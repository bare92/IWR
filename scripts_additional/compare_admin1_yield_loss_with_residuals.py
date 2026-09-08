#!/usr/bin/env python3
"""Compare gridded FAO33 yield-loss fractions with detrended Admin-1 yields.

The script performs four linked operations:

1. Matches products in ``residuals_data.csv`` to crop classes in the crop
   parameter CSV. Exact normalized names are used first, followed by explicit
   aliases (for example, ``Groundnuts (In Shell) -> groundnut``).
2. Reconstructs crop and merged yields from the residual dataset. Because the
   dataset contains separate trend rows for production and harvested area,
   expected yield is calculated as production trend / area trend. The merged
   yield across common crops is area-weighted by construction:

       merged yield = sum(production) / sum(harvested area)

3. Aggregates every annual yield-loss GeoTIFF to Admin-1 with a zonal weighted
   mean. Geographic rasters are latitude-area weighted. An optional multiband
   crop-fraction raster can be supplied; selected crop bands are summed inside
   the script and used to weight pixels by cropped area.
4. Writes auditable CSV tables and one two-panel PNG for every Admin-1 region.

The observed comparison variable is a signed detrended yield shortfall:

    1 - observed merged yield / expected merged yield

Positive values mean yield below trend; negative values mean yield above
trend. A positive-only version is also written for loss-magnitude diagnostics.

Example
-------
python compare_admin1_yield_loss_with_residuals.py \
  --residual-csv residuals_data.csv \
  --crop-config-csv CROPG_fractional.csv \
  --admin1-vector bfaadmbndaadm11msalbitos.zip \
  --yield-loss-dir /path/to/Seasonal_Yield_Loss_FAO33/Yield_loss_fraction \
  --output-dir /path/to/Yield_loss_validation_admin1 \
  --overwrite

Optional crop-area weighting from the original multiband crop-fraction raster:

  --crop-fraction-raster /path/to/CROPG_fractional.tif

By default, the script selects bands corresponding to model crops that have a
counterpart in the residual dataset. Use ``--crop-fraction-bands all`` to sum
every band, or e.g. ``--crop-fraction-bands 1-8`` for an explicit selection.
The legacy ``--crop-area-fraction-raster`` option remains available for an
already-summed single-band raster.

Required Python packages: numpy, pandas, matplotlib, geopandas, and rasterio.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import tempfile
import unicodedata
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


LOG = logging.getLogger("yield_loss_validation")


# These aliases are deliberately narrow. Substring/fuzzy matching is avoided
# because, for example, Bambara groundnut must not silently become groundnut.
DEFAULT_PRODUCT_ALIASES = {
    "groundnuts in shell": "groundnut",
    "sesame seed": "sesame",
    "sorghum red": "sorghum",
}


def normalize_label(value: object) -> str:
    """Return a stable key for crop and region label matching."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def safe_filename(value: object) -> str:
    """Return a portable filename stem."""

    stem = normalize_label(value).replace(" ", "_")
    return stem or "unnamed_admin1"


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Convert one column to numeric and fail with an informative message."""

    converted = pd.to_numeric(frame[column], errors="coerce")
    bad = converted.isna() & frame[column].notna()
    if bad.any():
        examples = frame.loc[bad, column].astype(str).head(5).tolist()
        raise ValueError(
            f"Column {column!r} contains non-numeric values; examples: {examples}"
        )
    return converted


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide where both values are finite and the denominator is positive."""

    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(num), np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0)
    result[valid] = num[valid] / den[valid]
    return pd.Series(result, index=numerator.index, dtype=float)


def add_yield_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add observed/trend yield and detrended shortfall fields."""

    out = frame.copy()
    out["yield_observed_t_ha"] = safe_divide(out["production_t"], out["area_ha"])
    out["yield_trend_t_ha"] = safe_divide(
        out["production_trend_t"], out["area_trend_ha"]
    )
    out["yield_residual_t_ha"] = (
        out["yield_observed_t_ha"] - out["yield_trend_t_ha"]
    )
    out["yield_anomaly_fraction"] = safe_divide(
        out["yield_residual_t_ha"], out["yield_trend_t_ha"]
    )
    out["yield_anomaly_pct"] = 100.0 * out["yield_anomaly_fraction"]
    out["observed_yield_shortfall_fraction_signed"] = -out[
        "yield_anomaly_fraction"
    ]
    out["observed_yield_loss_fraction_positive"] = out[
        "observed_yield_shortfall_fraction_signed"
    ].clip(lower=0.0)
    return out


def parse_alias_assignments(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``SOURCE=MODEL_CROP`` CLI assignments."""

    aliases = dict(DEFAULT_PRODUCT_ALIASES)
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --crop-alias {value!r}; expected SOURCE_PRODUCT=MODEL_CROP"
            )
        source, target = value.split("=", 1)
        source_key = normalize_label(source)
        target_key = normalize_label(target)
        if not source_key or not target_key:
            raise ValueError(
                f"Invalid --crop-alias {value!r}; both sides must be non-empty"
            )
        aliases[source_key] = target_key
    return aliases


def build_crop_match_table(
    residuals: pd.DataFrame,
    crop_config: pd.DataFrame,
    aliases: Mapping[str, str],
    excluded_products: Iterable[str],
) -> pd.DataFrame:
    """Build an explicit, auditable residual-product to model-crop crosswalk."""

    if "product" not in residuals.columns:
        raise ValueError("Residual CSV is missing required column 'product'")
    if "crop_name" not in crop_config.columns:
        raise ValueError("Crop configuration CSV is missing required column 'crop_name'")

    model_lookup: dict[str, str] = {}
    for crop in crop_config["crop_name"].dropna().astype(str):
        key = normalize_label(crop)
        if not key:
            continue
        if key in model_lookup and model_lookup[key] != crop:
            raise ValueError(
                "Crop configuration has labels that become duplicates after "
                f"normalization: {model_lookup[key]!r} and {crop!r}"
            )
        model_lookup[key] = crop

    if not model_lookup:
        raise ValueError("Crop configuration contains no usable crop names")

    excluded = {normalize_label(value) for value in excluded_products}
    records: list[dict[str, object]] = []
    matched_model_keys: set[str] = set()

    product_summary = (
        residuals.groupby("product", dropna=False)
        .agg(row_count=("product", "size"), first_year=("year", "min"), last_year=("year", "max"))
        .reset_index()
    )

    for row in product_summary.itertuples(index=False):
        product = str(row.product)
        product_key = normalize_label(product)
        model_key: str | None = None
        method = "unmatched"

        if product_key in excluded:
            method = "excluded_by_user"
        elif product_key in aliases:
            candidate = normalize_label(aliases[product_key])
            if candidate not in model_lookup:
                raise ValueError(
                    f"Alias target {aliases[product_key]!r} for residual product "
                    f"{product!r} is absent from crop configuration"
                )
            model_key = candidate
            method = "explicit_alias"
        elif product_key in model_lookup:
            model_key = product_key
            method = "exact_normalized"

        model_crop = model_lookup.get(model_key, "") if model_key else ""
        included = bool(model_crop) and method != "excluded_by_user"
        if included and model_key is not None:
            matched_model_keys.add(model_key)

        records.append(
            {
                "residual_product": product,
                "residual_product_normalized": product_key,
                "model_crop": model_crop,
                "match_method": method,
                "included": included,
                "residual_row_count": int(row.row_count),
                "first_year": int(row.first_year),
                "last_year": int(row.last_year),
            }
        )

    # Add model crops with no observational counterpart so the table explains
    # why they are absent from the merged observed yield.
    for model_key, model_crop in model_lookup.items():
        if model_key not in matched_model_keys:
            records.append(
                {
                    "residual_product": "",
                    "residual_product_normalized": "",
                    "model_crop": model_crop,
                    "match_method": "no_residual_counterpart",
                    "included": False,
                    "residual_row_count": 0,
                    "first_year": pd.NA,
                    "last_year": pd.NA,
                }
            )

    return pd.DataFrame.from_records(records).sort_values(
        ["included", "model_crop", "residual_product"],
        ascending=[False, True, True],
        kind="stable",
        ignore_index=True,
    )


def prepare_residual_yields(
    residual_csv: Path,
    crop_config_csv: Path,
    aliases: Mapping[str, str],
    excluded_products: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return crop-level yield, merged yield, and the crop crosswalk."""

    residuals = pd.read_csv(residual_csv)
    crop_config = pd.read_csv(crop_config_csv)

    required = {
        "year",
        "region",
        "product",
        "production",
        "area",
        "trend",
        "residuals",
        "residuals_pct",
        "variable",
    }
    missing = sorted(required - set(residuals.columns))
    if missing:
        raise ValueError(f"Residual CSV is missing required columns: {missing}")

    residuals = residuals.copy()
    for column in ["year", "production", "area", "trend", "residuals", "residuals_pct"]:
        residuals[column] = numeric_column(residuals, column)
    residuals["year"] = residuals["year"].astype(int)
    residuals["_variable_key"] = residuals["variable"].map(normalize_label)

    unexpected_variables = sorted(
        set(residuals["_variable_key"].dropna()) - {"production", "area"}
    )
    if unexpected_variables:
        raise ValueError(
            "Residual CSV contains unexpected variable labels: "
            f"{unexpected_variables}; expected only production and area"
        )

    match_table = build_crop_match_table(
        residuals, crop_config, aliases, excluded_products
    )
    product_to_crop = {
        str(row.residual_product): str(row.model_crop)
        for row in match_table.itertuples(index=False)
        if bool(row.included)
    }
    if not product_to_crop:
        raise ValueError("No residual products match crops in the crop configuration")

    keys = ["year", "region", "product"]
    production_rows = residuals.loc[
        residuals["_variable_key"].eq("production")
    ].copy()
    area_rows = residuals.loc[residuals["_variable_key"].eq("area")].copy()

    for label, frame in [("production", production_rows), ("area", area_rows)]:
        duplicate = frame.duplicated(keys, keep=False)
        if duplicate.any():
            examples = frame.loc[duplicate, keys].head(5).to_dict("records")
            raise ValueError(
                f"Duplicate {label} residual rows for year/region/product; "
                f"examples: {examples}"
            )

    production_rows = production_rows[
        keys + ["production", "area", "trend", "residuals", "residuals_pct"]
    ].rename(
        columns={
            "production": "production_from_production_row",
            "area": "area_from_production_row",
            "trend": "production_trend_t",
            "residuals": "production_residual_t",
            "residuals_pct": "production_residual_pct",
        }
    )
    area_rows = area_rows[
        keys + ["production", "area", "trend", "residuals", "residuals_pct"]
    ].rename(
        columns={
            "production": "production_from_area_row",
            "area": "area_from_area_row",
            "trend": "area_trend_ha",
            "residuals": "area_residual_ha",
            "residuals_pct": "area_residual_pct",
        }
    )

    paired = production_rows.merge(
        area_rows, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    unpaired = paired["_merge"].ne("both")
    if unpaired.any():
        examples = paired.loc[unpaired, keys + ["_merge"]].head(5).to_dict("records")
        raise ValueError(
            "Production and area residual rows are not paired one-to-one; "
            f"examples: {examples}"
        )

    if not np.allclose(
        paired["production_from_production_row"],
        paired["production_from_area_row"],
        equal_nan=True,
    ):
        raise ValueError("Raw production differs between production and area rows")
    if not np.allclose(
        paired["area_from_production_row"],
        paired["area_from_area_row"],
        equal_nan=True,
    ):
        raise ValueError("Raw area differs between production and area rows")

    paired = paired.rename(
        columns={
            "production_from_production_row": "production_t",
            "area_from_production_row": "area_ha",
        }
    ).drop(
        columns=[
            "production_from_area_row",
            "area_from_area_row",
            "_merge",
        ]
    )
    paired["model_crop"] = paired["product"].map(product_to_crop)
    matched = paired.loc[paired["model_crop"].notna()].copy()

    def joined_unique(values: pd.Series) -> str:
        return " | ".join(sorted({str(value) for value in values if pd.notna(value)}))

    crop_level = (
        matched.groupby(["year", "region", "model_crop"], as_index=False)
        .agg(
            production_t=("production_t", "sum"),
            area_ha=("area_ha", "sum"),
            production_trend_t=("production_trend_t", "sum"),
            area_trend_ha=("area_trend_ha", "sum"),
            production_residual_t=("production_residual_t", "sum"),
            area_residual_ha=("area_residual_ha", "sum"),
            n_residual_products=("product", "nunique"),
            residual_products=("product", joined_unique),
        )
        .sort_values(["region", "model_crop", "year"], kind="stable")
        .reset_index(drop=True)
    )
    crop_level = add_yield_fields(crop_level)

    merged = (
        crop_level.groupby(["year", "region"], as_index=False)
        .agg(
            production_t=("production_t", "sum"),
            area_ha=("area_ha", "sum"),
            production_trend_t=("production_trend_t", "sum"),
            area_trend_ha=("area_trend_ha", "sum"),
            production_residual_t=("production_residual_t", "sum"),
            area_residual_ha=("area_residual_ha", "sum"),
            n_model_crops=("model_crop", "nunique"),
            model_crops_present=("model_crop", joined_unique),
            n_residual_products=("n_residual_products", "sum"),
        )
        .sort_values(["region", "year"], kind="stable")
        .reset_index(drop=True)
    )
    merged = add_yield_fields(merged)

    return crop_level, merged, match_table


@contextmanager
def resolved_vector_path(path: Path, layer: str | None) -> Iterator[tuple[Path, str | None]]:
    """Yield a local vector path, safely extracting a zipped shapefile."""

    if path.suffix.lower() != ".zip":
        yield path, layer
        return

    with tempfile.TemporaryDirectory(prefix="admin1_vector_") as temp_name:
        temp_dir = Path(temp_name).resolve()
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                destination = (temp_dir / member.filename).resolve()
                try:
                    destination.relative_to(temp_dir)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsafe path in Admin-1 ZIP archive: {member.filename!r}"
                    ) from exc
                archive.extract(member, temp_dir)

        shapefiles = sorted(temp_dir.rglob("*.shp"))
        if layer:
            requested = normalize_label(Path(layer).stem)
            shapefiles = [
                candidate
                for candidate in shapefiles
                if normalize_label(candidate.stem) == requested
            ]
        if len(shapefiles) != 1:
            found = [candidate.name for candidate in shapefiles]
            raise ValueError(
                "Expected exactly one Admin-1 shapefile in ZIP"
                + (f" matching --admin-layer {layer!r}" if layer else "")
                + f"; found {found}"
            )
        yield shapefiles[0], None


def import_geospatial_packages():
    """Import geospatial dependencies lazily and provide a concise error."""

    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.warp import Resampling, reproject
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing geospatial dependency. Activate the IWR environment and "
            "install/enable geopandas and rasterio before running this script. "
            f"Original error: {exc}"
        ) from exc
    return gpd, rasterio, geometry_mask, Resampling, reproject


def load_admin1(
    vector_path: Path,
    layer: str | None,
    name_field: str,
):
    """Read, validate, and dissolve Admin-1 geometries by name."""

    gpd, _, _, _, _ = import_geospatial_packages()
    with resolved_vector_path(vector_path, layer) as (resolved_path, resolved_layer):
        kwargs = {"layer": resolved_layer} if resolved_layer else {}
        admin = gpd.read_file(resolved_path, **kwargs)

    if admin.empty:
        raise ValueError(f"Admin-1 vector contains no features: {vector_path}")
    if admin.crs is None:
        raise ValueError(f"Admin-1 vector has no CRS: {vector_path}")
    if name_field not in admin.columns:
        raise ValueError(
            f"Admin name field {name_field!r} not found. Available fields: "
            f"{sorted(column for column in admin.columns if column != admin.geometry.name)}"
        )

    admin = admin[[name_field, admin.geometry.name]].copy()
    admin[name_field] = admin[name_field].astype(str).str.strip()
    admin = admin.loc[admin[name_field].ne("") & admin.geometry.notna()].copy()
    admin = admin.loc[~admin.geometry.is_empty].copy()
    if admin.empty:
        raise ValueError("Admin-1 vector has no named, non-empty geometries")

    try:
        admin.geometry = admin.geometry.make_valid()
    except (AttributeError, NotImplementedError):
        admin.geometry = admin.geometry.buffer(0)

    admin = admin.dissolve(by=name_field, as_index=False)
    admin["_region_key"] = admin[name_field].map(normalize_label)
    duplicates = admin["_region_key"].duplicated(keep=False)
    if duplicates.any():
        names = admin.loc[duplicates, name_field].tolist()
        raise ValueError(
            "Admin names are ambiguous after normalization; rename or select a "
            f"different field: {names}"
        )
    return admin


def map_residual_regions(
    crop_level: pd.DataFrame,
    merged: pd.DataFrame,
    admin,
    admin_name_field: str,
    allow_unmatched: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map residual region labels to canonical vector Admin-1 names."""

    admin_lookup = dict(zip(admin["_region_key"], admin[admin_name_field]))
    residual_names = sorted(set(merged["region"].astype(str)))
    mapping_records: list[dict[str, object]] = []
    for region in residual_names:
        key = normalize_label(region)
        canonical = admin_lookup.get(key)
        mapping_records.append(
            {
                "residual_region": region,
                "residual_region_normalized": key,
                "admin1_name": canonical if canonical is not None else "",
                "matched": canonical is not None,
            }
        )
    region_crosswalk = pd.DataFrame.from_records(mapping_records)
    unmatched = region_crosswalk.loc[~region_crosswalk["matched"], "residual_region"].tolist()
    if unmatched and not allow_unmatched:
        raise ValueError(
            "Residual regions do not match the Admin-1 vector after normalized "
            f"matching: {unmatched}. Use --allow-unmatched-regions only if these "
            "regions should be excluded from the comparison."
        )
    if unmatched:
        LOG.warning("Excluding unmatched residual regions: %s", unmatched)

    lookup = {
        row.residual_region: row.admin1_name
        for row in region_crosswalk.itertuples(index=False)
        if bool(row.matched)
    }

    def apply_mapping(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.insert(2, "admin1_name", out["region"].map(lookup))
        out = out.loc[out["admin1_name"].notna()].copy()
        return out

    return apply_mapping(crop_level), apply_mapping(merged), region_crosswalk


def discover_rasters(
    directory: Path,
    pattern: str,
    start_year: int | None,
    end_year: int | None,
) -> list[tuple[int, Path]]:
    """Discover annual rasters and parse the final four-digit year."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Yield-loss directory not found: {directory}")
    year_pattern = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
    found: list[tuple[int, Path]] = []
    for path in sorted(directory.glob(pattern)):
        matches = year_pattern.findall(path.stem)
        if not matches:
            LOG.warning("Skipping raster with no four-digit year: %s", path.name)
            continue
        year = int(matches[-1])
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        found.append((year, path))

    if not found:
        raise FileNotFoundError(
            f"No annual rasters matched {pattern!r} in {directory} for the "
            "requested year range"
        )
    years = [year for year, _ in found]
    duplicate_years = sorted({year for year in years if years.count(year) > 1})
    if duplicate_years:
        raise ValueError(
            f"Multiple yield-loss rasters found for years {duplicate_years}; "
            "narrow --raster-pattern"
        )
    return sorted(found)


def grid_key(dataset, crs_override=None) -> tuple[object, ...]:
    """Return a hashable raster grid signature."""

    crs_obj = crs_override if crs_override is not None else dataset.crs
    crs_text = crs_obj.to_wkt() if crs_obj is not None else ""
    return (
        crs_text,
        int(dataset.width),
        int(dataset.height),
        tuple(float(value) for value in dataset.transform),
    )


def pixel_area_relative_weights(dataset, mode: str, crs=None) -> np.ndarray:
    """Return equal or latitude-adjusted relative pixel-area weights."""

    shape = (dataset.height, dataset.width)
    crs_obj = crs if crs is not None else dataset.crs
    if mode == "equal" or crs_obj is None or not crs_obj.is_geographic:
        return np.ones(shape, dtype=np.float64)

    rows, columns = np.indices(shape, dtype=np.float64)
    transform = dataset.transform
    latitudes = (
        transform.f
        + transform.d * (columns + 0.5)
        + transform.e * (rows + 0.5)
    )
    weights = np.cos(np.deg2rad(latitudes))
    weights[~np.isfinite(weights) | (weights <= 0)] = np.nan
    return weights


def parse_band_expression(expression: str, band_count: int) -> list[int]:
    """Parse a 1-based band expression such as ``1-8,10``."""

    bands: list[int] = []
    for token in expression.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(
                    f"Invalid crop-fraction band range {token!r}; use e.g. 1-8,10"
                )
            start, end = (int(part.strip()) for part in parts)
            if start > end:
                raise ValueError(f"Invalid descending crop-fraction range: {token!r}")
            bands.extend(range(start, end + 1))
        elif token.isdigit():
            bands.append(int(token))
        else:
            raise ValueError(
                f"Invalid crop-fraction band token {token!r}; use common, all, or e.g. 1-8"
            )

    bands = list(dict.fromkeys(bands))
    if not bands:
        raise ValueError("Crop-fraction band selection is empty")
    invalid = [band for band in bands if not 1 <= band <= band_count]
    if invalid:
        raise ValueError(
            f"Crop-fraction band(s) {invalid} are invalid for a raster with "
            f"{band_count} band(s)"
        )
    return bands


def resolve_crop_fraction_band_selection(
    raster_path: Path,
    band_specification: str,
    crop_config_csv: Path,
    crop_match_table: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve multiband crop fractions to explicit 1-based band numbers.

    ``common`` selects model crops that have a residual-data counterpart.
    Raster band descriptions are preferred when they identify every requested
    crop uniquely; otherwise the crop configuration row order defines the band
    order. ``all`` selects every raster band, and an explicit expression such
    as ``1-8,10`` selects those exact bands.
    """

    _, rasterio, _, _, _ = import_geospatial_packages()
    crop_config = pd.read_csv(crop_config_csv)
    if "crop_name" not in crop_config.columns:
        raise ValueError("Crop configuration CSV is missing required column 'crop_name'")

    configured_crops = [
        str(value).strip()
        for value in crop_config["crop_name"].dropna()
        if str(value).strip()
    ]
    configured_keys = [normalize_label(value) for value in configured_crops]
    duplicate_keys = sorted(
        {key for key in configured_keys if configured_keys.count(key) > 1}
    )
    if duplicate_keys:
        raise ValueError(
            "Crop configuration contains duplicate normalized crop names: "
            f"{duplicate_keys}"
        )

    common_keys = {
        normalize_label(value)
        for value in crop_match_table.loc[
            crop_match_table["included"], "model_crop"
        ].astype(str)
    }

    with rasterio.open(raster_path) as source:
        band_count = source.count
        descriptions = list(source.descriptions)

    specification = band_specification.strip().lower()
    if band_count == 1 and specification in {"common", "all"}:
        return pd.DataFrame.from_records(
            [
                {
                    "band": 1,
                    "crop_name": descriptions[0] or "pre_summed_crop_fraction",
                    "crop_name_normalized": normalize_label(
                        descriptions[0] or "pre_summed_crop_fraction"
                    ),
                    "selection_source": "single_band_pre_summed_raster",
                }
            ]
        )

    description_lookup: dict[str, list[tuple[int, str]]] = {}
    for band, description in enumerate(descriptions, start=1):
        if description and normalize_label(description):
            description_lookup.setdefault(normalize_label(description), []).append(
                (band, str(description))
            )

    if specification == "common":
        if not common_keys:
            raise ValueError("No common model crops are available for band selection")

        descriptions_are_complete = all(
            key in description_lookup and len(description_lookup[key]) == 1
            for key in common_keys
        )
        if descriptions_are_complete:
            records = []
            for key in configured_keys:
                if key not in common_keys:
                    continue
                band, description = description_lookup[key][0]
                records.append(
                    {
                        "band": band,
                        "crop_name": description,
                        "crop_name_normalized": key,
                        "selection_source": "raster_band_description",
                    }
                )
            return pd.DataFrame.from_records(records).sort_values(
                "band", ignore_index=True
            )

        requested_positions = [
            index
            for index, key in enumerate(configured_keys, start=1)
            if key in common_keys
        ]
        if not requested_positions:
            raise ValueError(
                "None of the common crops can be located in the crop configuration"
            )
        if max(requested_positions) > band_count:
            raise ValueError(
                "The multiband crop-fraction raster has fewer bands than required "
                "by the common crops in crop configuration row order: required "
                f"band {max(requested_positions)}, raster has {band_count} band(s)"
            )
        if any(descriptions):
            LOG.warning(
                "Raster band descriptions do not uniquely identify every common "
                "crop; using crop configuration row order as the band order"
            )
        return pd.DataFrame.from_records(
            [
                {
                    "band": band,
                    "crop_name": configured_crops[band - 1],
                    "crop_name_normalized": configured_keys[band - 1],
                    "selection_source": "crop_config_row_order",
                }
                for band in requested_positions
            ]
        )

    if specification == "all":
        selected_bands = list(range(1, band_count + 1))
        selection_source = "all_raster_bands"
    else:
        selected_bands = parse_band_expression(specification, band_count)
        selection_source = "explicit_band_expression"

    records = []
    for band in selected_bands:
        description = descriptions[band - 1]
        if description and normalize_label(description):
            crop_name = str(description)
        elif band <= len(configured_crops):
            crop_name = configured_crops[band - 1]
        else:
            crop_name = f"band_{band}"
        records.append(
            {
                "band": band,
                "crop_name": crop_name,
                "crop_name_normalized": normalize_label(crop_name),
                "selection_source": selection_source,
            }
        )
    return pd.DataFrame.from_records(records)


def aligned_crop_weights(
    target,
    target_crs,
    weight_path: Path,
    weight_bands: Sequence[int],
    rasterio,
    Resampling,
    reproject,
) -> np.ndarray:
    """Sum selected crop-fraction bands and align them to a target grid."""

    if not weight_bands:
        raise ValueError("At least one crop-fraction band must be selected")

    with rasterio.open(weight_path) as source:
        invalid_bands = [
            band for band in weight_bands if not 1 <= band <= source.count
        ]
        if invalid_bands:
            raise ValueError(
                f"Crop-fraction band(s) {invalid_bands} are invalid for "
                f"{weight_path} with {source.count} band(s)"
            )

        effective_source_crs = source.crs
        grids_match_without_crs = (
            source.width == target.width
            and source.height == target.height
            and source.transform.almost_equals(target.transform)
        )
        if effective_source_crs is None:
            if target_crs is None or not grids_match_without_crs:
                raise ValueError(
                    "Crop-fraction raster has no CRS and cannot be safely aligned "
                    f"to the yield-loss grid: {weight_path}"
                )
            effective_source_crs = target_crs
            LOG.warning(
                "Crop-fraction raster has no CRS but exactly matches the yield-loss "
                "grid; assuming target CRS %s",
                target_crs,
            )

        stacked = source.read(list(weight_bands), masked=True).astype(np.float64)
        raw_values = np.asarray(stacked, dtype=np.float64)
        band_masks = np.ma.getmaskarray(stacked)
        valid_band_values = (
            ~band_masks & np.isfinite(raw_values) & (raw_values >= 0)
        )
        source_values = np.sum(
            np.where(valid_band_values, raw_values, 0.0), axis=0, dtype=np.float64
        )
        source_values[~np.any(valid_band_values, axis=0)] = np.nan

        same_grid = (
            effective_source_crs == target_crs
            and grids_match_without_crs
        )
        if same_grid:
            weights = source_values
        else:
            weights = np.full((target.height, target.width), np.nan, dtype=np.float64)
            reproject(
                source=source_values,
                destination=weights,
                src_transform=source.transform,
                src_crs=effective_source_crs,
                src_nodata=np.nan,
                dst_transform=target.transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
    weights[~np.isfinite(weights) | (weights <= 0)] = np.nan
    return weights


def aggregate_yield_loss_rasters(
    rasters: Sequence[tuple[int, Path]],
    admin,
    admin_name_field: str,
    raster_band: int,
    spatial_weighting: str,
    crop_fraction_raster: Path | None,
    crop_fraction_bands: Sequence[int],
    all_touched: bool,
    loss_scale: float,
    valid_loss_min: float,
    valid_loss_max: float,
    min_valid_pixels: int,
) -> pd.DataFrame:
    """Aggregate annual yield-loss rasters to Admin-1 weighted means."""

    gpd, rasterio, geometry_mask, Resampling, reproject = import_geospatial_packages()
    del gpd  # Imported for dependency validation; the GeoDataFrame is supplied.

    geometry_cache: dict[str, object] = {}
    area_weight_cache: dict[tuple[object, ...], np.ndarray] = {}
    crop_weight_cache: dict[tuple[object, ...], np.ndarray] = {}
    records: list[dict[str, object]] = []
    warned_missing_raster_crs = False

    for year, raster_path in rasters:
        LOG.info("Aggregating %s", raster_path.name)
        with rasterio.open(raster_path) as source:
            if source.crs is None:
                if admin.crs is None:
                    raise ValueError(
                        "Yield-loss raster has no CRS and Admin-1 vector CRS is "
                        f"also missing: {raster_path}"
                    )
                source_crs = admin.crs
                if not warned_missing_raster_crs:
                    LOG.warning(
                        "Yield-loss rasters have no CRS. Assuming Admin-1 CRS for "
                        "all such rasters: %s",
                        source_crs,
                    )
                    warned_missing_raster_crs = True
            else:
                source_crs = source.crs
            if not 1 <= raster_band <= source.count:
                raise ValueError(
                    f"Raster band {raster_band} is invalid for {raster_path} "
                    f"with {source.count} band(s)"
                )

            crs_key = source_crs.to_wkt()
            if crs_key not in geometry_cache:
                if source.crs is None:
                    geometry_cache[crs_key] = admin
                else:
                    geometry_cache[crs_key] = admin.to_crs(source_crs)
            admin_on_grid = geometry_cache[crs_key]

            key = grid_key(source, source_crs)
            if key not in area_weight_cache:
                area_weight_cache[key] = pixel_area_relative_weights(
                    source, spatial_weighting, source_crs
                )
            spatial_weights = area_weight_cache[key]

            if crop_fraction_raster is not None:
                if key not in crop_weight_cache:
                    crop_weight_cache[key] = aligned_crop_weights(
                        source,
                        source_crs,
                        crop_fraction_raster,
                        crop_fraction_bands,
                        rasterio,
                        Resampling,
                        reproject,
                    )
                weights = spatial_weights * crop_weight_cache[key]
                band_label = ",".join(str(band) for band in crop_fraction_bands)
                weighting_label = (
                    f"{spatial_weighting}_x_sum_crop_fraction_bands_{band_label}"
                )
            else:
                weights = spatial_weights
                weighting_label = spatial_weighting

            masked_data = source.read(raster_band, masked=True)
            values = np.asarray(masked_data, dtype=np.float64) * loss_scale
            source_mask = np.ma.getmaskarray(masked_data)

            for _, feature in admin_on_grid.iterrows():
                region = str(feature[admin_name_field])
                geometry = feature.geometry
                inside = geometry_mask(
                    [geometry.__geo_interface__],
                    out_shape=values.shape,
                    transform=source.transform,
                    invert=True,
                    all_touched=all_touched,
                )
                inside_count = int(np.count_nonzero(inside))
                finite_source = inside & ~source_mask & np.isfinite(values)
                out_of_range = finite_source & (
                    (values < valid_loss_min) | (values > valid_loss_max)
                )
                valid = (
                    finite_source
                    & ~out_of_range
                    & np.isfinite(weights)
                    & (weights > 0)
                )
                valid_count = int(np.count_nonzero(valid))
                out_of_range_count = int(np.count_nonzero(out_of_range))

                if valid_count >= min_valid_pixels:
                    selected_weights = weights[valid]
                    weight_sum = float(np.sum(selected_weights))
                    mean_loss = float(
                        np.sum(values[valid] * selected_weights) / weight_sum
                    )
                    min_loss = float(np.min(values[valid]))
                    max_loss = float(np.max(values[valid]))
                else:
                    weight_sum = math.nan
                    mean_loss = math.nan
                    min_loss = math.nan
                    max_loss = math.nan

                records.append(
                    {
                        "year": int(year),
                        "admin1_name": region,
                        "modeled_yield_loss_fraction": mean_loss,
                        "valid_pixel_count": valid_count,
                        "inside_polygon_pixel_count": inside_count,
                        "valid_pixel_fraction": (
                            valid_count / inside_count if inside_count else math.nan
                        ),
                        "out_of_range_pixel_count": out_of_range_count,
                        "aggregation_weight_sum": weight_sum,
                        "minimum_valid_loss_fraction": min_loss,
                        "maximum_valid_loss_fraction": max_loss,
                        "aggregation_weighting": weighting_label,
                        "crop_fraction_bands_used": (
                            ",".join(str(band) for band in crop_fraction_bands)
                            if crop_fraction_raster is not None
                            else ""
                        ),
                        "source_raster": str(raster_path),
                    }
                )

    summary = pd.DataFrame.from_records(records).sort_values(
        ["admin1_name", "year"], kind="stable", ignore_index=True
    )
    total_out_of_range = int(summary["out_of_range_pixel_count"].sum())
    if total_out_of_range:
        LOG.warning(
            "Excluded %d Admin-1/year pixel occurrences outside [%g, %g] after "
            "applying --loss-scale=%g. Check nodata metadata and units if this is unexpected.",
            total_out_of_range,
            valid_loss_min,
            valid_loss_max,
            loss_scale,
        )
    if int(summary["valid_pixel_count"].sum()) == 0:
        raise ValueError(
            "No valid yield-loss pixels remained after spatial masking and the "
            f"[{valid_loss_min}, {valid_loss_max}] range filter. Check raster/vector "
            "CRSs, nodata metadata, and --loss-scale (use 0.01 for percent maps)."
        )
    return summary


def correlation(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation with finite/constant checks."""

    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 3:
        return math.nan
    x_values = x_values[valid]
    y_values = y_values[valid]
    if np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def spearman_correlation(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation without requiring scipy."""

    pair = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        return math.nan
    return correlation(pair["x"].rank(method="average"), pair["y"].rank(method="average"))


def slope_per_decade(year: pd.Series, values: pd.Series) -> float:
    """OLS linear slope per decade with finite/constant checks."""

    x = pd.to_numeric(year, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(x[valid]) == 0:
        return math.nan
    centered_year = x[valid] - np.mean(x[valid])
    slope = np.polyfit(centered_year, y[valid], 1)[0]
    return float(10.0 * slope)


def build_comparison(
    map_summary: pd.DataFrame,
    merged_yield: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer-join modeled and observed series and calculate region statistics."""

    observed_columns = [
        "year",
        "admin1_name",
        "region",
        "production_t",
        "area_ha",
        "production_trend_t",
        "area_trend_ha",
        "production_residual_t",
        "area_residual_ha",
        "n_model_crops",
        "model_crops_present",
        "n_residual_products",
        "yield_observed_t_ha",
        "yield_trend_t_ha",
        "yield_residual_t_ha",
        "yield_anomaly_fraction",
        "yield_anomaly_pct",
        "observed_yield_shortfall_fraction_signed",
        "observed_yield_loss_fraction_positive",
    ]
    comparison = map_summary.merge(
        merged_yield[observed_columns],
        on=["year", "admin1_name"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["admin1_name", "year"], kind="stable", ignore_index=True)
    comparison["both_sources_available"] = comparison[
        ["modeled_yield_loss_fraction", "observed_yield_shortfall_fraction_signed"]
    ].notna().all(axis=1)
    comparison["modeled_minus_observed_positive_loss"] = (
        comparison["modeled_yield_loss_fraction"]
        - comparison["observed_yield_loss_fraction_positive"]
    )

    records: list[dict[str, object]] = []
    for region, group in comparison.groupby("admin1_name", sort=True):
        overlap = group.loc[group["both_sources_available"]].copy()
        difference = overlap["modeled_minus_observed_positive_loss"].dropna()
        records.append(
            {
                "admin1_name": region,
                "n_map_years": int(group["modeled_yield_loss_fraction"].notna().sum()),
                "n_observed_years": int(group["yield_observed_t_ha"].notna().sum()),
                "n_overlap_years": int(len(overlap)),
                "overlap_start_year": (
                    int(overlap["year"].min()) if len(overlap) else pd.NA
                ),
                "overlap_end_year": (
                    int(overlap["year"].max()) if len(overlap) else pd.NA
                ),
                "pearson_r_model_vs_observed_signed_shortfall": correlation(
                    overlap["modeled_yield_loss_fraction"],
                    overlap["observed_yield_shortfall_fraction_signed"],
                ),
                "spearman_rho_model_vs_observed_signed_shortfall": spearman_correlation(
                    overlap["modeled_yield_loss_fraction"],
                    overlap["observed_yield_shortfall_fraction_signed"],
                ),
                "rmse_model_vs_observed_positive_loss": (
                    float(np.sqrt(np.mean(np.square(difference))))
                    if len(difference)
                    else math.nan
                ),
                "mean_bias_model_minus_observed_positive_loss": (
                    float(difference.mean()) if len(difference) else math.nan
                ),
                "mean_modeled_yield_loss_fraction": float(
                    group["modeled_yield_loss_fraction"].mean()
                ),
                "mean_observed_signed_shortfall_fraction": float(
                    group["observed_yield_shortfall_fraction_signed"].mean()
                ),
                "modeled_loss_slope_fraction_per_decade": slope_per_decade(
                    group["year"], group["modeled_yield_loss_fraction"]
                ),
                "observed_shortfall_slope_fraction_per_decade": slope_per_decade(
                    group["year"],
                    group["observed_yield_shortfall_fraction_signed"],
                ),
                "observed_yield_slope_t_ha_per_decade": slope_per_decade(
                    group["year"], group["yield_observed_t_ha"]
                ),
                "expected_yield_slope_t_ha_per_decade": slope_per_decade(
                    group["year"], group["yield_trend_t_ha"]
                ),
            }
        )

    statistics = pd.DataFrame.from_records(records).sort_values(
        "admin1_name", kind="stable", ignore_index=True
    )
    return comparison, statistics


def finite_line_fit(year: pd.Series, values: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Return x/y coordinates for a linear trend line."""

    x = pd.to_numeric(year, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(x[valid]) == 0:
        return None
    fit_x = np.array([np.min(x[valid]), np.max(x[valid])], dtype=float)
    centered = x[valid] - np.mean(x[valid])
    slope, intercept = np.polyfit(centered, y[valid], 1)
    fit_y = intercept + slope * (fit_x - np.mean(x[valid]))
    return fit_x, fit_y


def plot_admin1_comparisons(
    comparison: pd.DataFrame,
    statistics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Create one two-panel validation plot for every Admin-1 region."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_lookup = statistics.set_index("admin1_name").to_dict("index")
    written: list[Path] = []

    for region, group in comparison.groupby("admin1_name", sort=True):
        group = group.sort_values("year")
        stat = statistics_lookup.get(region, {})

        fig, (loss_axis, yield_axis) = plt.subplots(
            2,
            1,
            figsize=(12.0, 8.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1.0]},
        )
        fig.suptitle(
            f"{region}: modeled FAO33 loss and observed merged yield",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )

        loss_axis.axhline(0.0, color="#777777", linewidth=0.9, zorder=1)
        loss_axis.plot(
            group["year"],
            group["modeled_yield_loss_fraction"],
            color="#c43c39",
            marker="o",
            markersize=3.7,
            linewidth=1.8,
            label="Modeled FAO33 yield loss (Admin-1 mean)",
            zorder=3,
        )
        loss_axis.plot(
            group["year"],
            group["observed_yield_shortfall_fraction_signed"],
            color="#245b8a",
            marker="s",
            markersize=3.1,
            linewidth=1.35,
            label="Observed detrended yield shortfall (signed)",
            zorder=2,
        )

        for column, color in [
            ("modeled_yield_loss_fraction", "#c43c39"),
            ("observed_yield_shortfall_fraction_signed", "#245b8a"),
        ]:
            fit = finite_line_fit(group["year"], group[column])
            if fit is not None:
                loss_axis.plot(
                    fit[0], fit[1], color=color, linestyle="--", linewidth=1.0, alpha=0.75
                )

        loss_axis.set_ylabel("Yield-loss / shortfall fraction")
        loss_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        loss_axis.grid(True, axis="both", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        loss_axis.legend(loc="best", frameon=False, fontsize=9)

        pearson = stat.get("pearson_r_model_vs_observed_signed_shortfall", math.nan)
        spearman = stat.get("spearman_rho_model_vs_observed_signed_shortfall", math.nan)
        n_overlap = int(stat.get("n_overlap_years", 0) or 0)
        correlation_text = (
            f"Overlap n={n_overlap} | Pearson r={pearson:.2f} | Spearman rho={spearman:.2f}"
            if np.isfinite(pearson) and np.isfinite(spearman)
            else f"Overlap n={n_overlap} | correlation unavailable"
        )
        loss_axis.text(
            0.01,
            0.02,
            correlation_text,
            transform=loss_axis.transAxes,
            fontsize=9,
            color="#333333",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.5},
        )

        yield_axis.plot(
            group["year"],
            group["yield_observed_t_ha"],
            color="#202020",
            marker="o",
            markersize=3.4,
            linewidth=1.6,
            label="Observed merged yield",
        )
        yield_axis.plot(
            group["year"],
            group["yield_trend_t_ha"],
            color="#d58a20",
            linestyle="--",
            linewidth=2.0,
            label="Expected yield from production/area trends",
        )
        yield_axis.set_ylabel("Merged yield (t/ha)")
        yield_axis.set_xlabel("Year")
        yield_axis.grid(True, axis="both", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        yield_axis.legend(loc="best", frameon=False, fontsize=9)

        years = pd.to_numeric(group["year"], errors="coerce").dropna()
        if len(years):
            first_year = int(years.min())
            last_year = int(years.max())
            tick_start = int(math.ceil(first_year / 5.0) * 5)
            ticks = list(range(tick_start, last_year + 1, 5))
            if first_year not in ticks:
                ticks.insert(0, first_year)
            if last_year not in ticks:
                ticks.append(last_year)
            yield_axis.set_xticks(sorted(set(ticks)))
            yield_axis.set_xlim(first_year - 0.6, last_year + 0.6)

        fig.text(
            0.5,
            0.012,
            "Common-crop yield = sum(production) / sum(harvested area). "
            "Positive observed shortfall means yield below its detrended expectation; "
            "negative values mean yield above expectation.",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#4a4a4a",
        )
        fig.tight_layout(rect=(0.025, 0.05, 0.985, 0.96))

        output_path = output_dir / f"yield_loss_vs_residual_yield_{safe_filename(region)}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(output_path)

    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate annual yield-loss rasters to Admin-1 and compare them "
            "with merged, detrended yields for crops common to the model and "
            "the residual dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--residual-csv", required=True, type=Path)
    parser.add_argument("--crop-config-csv", required=True, type=Path)
    parser.add_argument(
        "--admin1-vector",
        required=True,
        type=Path,
        help="Admin-1 ZIP shapefile or any vector file readable by geopandas",
    )
    parser.add_argument(
        "--admin-layer",
        help="Layer name for a multi-layer vector, or shapefile stem inside a ZIP",
    )
    parser.add_argument("--admin-name-field", default="admin1Name")
    parser.add_argument("--yield-loss-dir", required=True, type=Path)
    parser.add_argument(
        "--raster-pattern",
        default="yield_loss_fraction_season1_*.tif",
        help="Glob relative to --yield-loss-dir; the final four-digit year is parsed",
    )
    parser.add_argument("--raster-band", type=int, default=1)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--crop-alias",
        action="append",
        default=[],
        metavar="SOURCE_PRODUCT=MODEL_CROP",
        help="Add or override an explicit crop match; repeat as needed",
    )
    parser.add_argument(
        "--exclude-residual-product",
        action="append",
        default=[],
        help="Residual product to exclude from the common-crop aggregate; repeat as needed",
    )
    parser.add_argument(
        "--allow-unmatched-regions",
        action="store_true",
        help="Exclude residual regions absent from the Admin-1 vector instead of failing",
    )
    parser.add_argument(
        "--spatial-weighting",
        choices=["pixel-area", "equal"],
        default="pixel-area",
        help="For geographic rasters, pixel-area applies cosine(latitude) weighting",
    )
    crop_fraction_group = parser.add_mutually_exclusive_group()
    crop_fraction_group.add_argument(
        "--crop-fraction-raster",
        type=Path,
        help=(
            "Optional multiband crop-fraction raster. Selected bands are summed "
            "internally and multiply the spatial weights, making the Admin-1 "
            "modeled loss crop-area weighted"
        ),
    )
    crop_fraction_group.add_argument(
        "--crop-area-fraction-raster",
        type=Path,
        help=(
            "Legacy input for an already-summed single-band crop-area fraction "
            "raster; use --crop-fraction-raster for the multiband source"
        ),
    )
    parser.add_argument(
        "--crop-fraction-bands",
        default="common",
        metavar="common|all|BANDS",
        help=(
            "Bands to sum from --crop-fraction-raster. 'common' selects crop "
            "classes matched to residual data using band descriptions or crop "
            "configuration row order; 'all' uses every band; explicit examples: "
            "1-8 or 1,2,4-7"
        ),
    )
    parser.add_argument(
        "--crop-area-fraction-band",
        type=int,
        default=1,
        help="Band used only with legacy --crop-area-fraction-raster",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Include every pixel touched by an Admin-1 polygon, not only center pixels",
    )
    parser.add_argument(
        "--loss-scale",
        type=float,
        default=1.0,
        help="Multiply raster values by this factor (use 0.01 if maps store percent)",
    )
    parser.add_argument("--valid-loss-min", type=float, default=0.0)
    parser.add_argument("--valid-loss-max", type=float, default=1.0)
    parser.add_argument("--min-valid-pixels", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow known output files in a non-empty output directory to be overwritten",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for label in ["residual_csv", "crop_config_csv", "admin1_vector"]:
        path = getattr(args, label)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    for fraction_path in [
        args.crop_fraction_raster,
        args.crop_area_fraction_raster,
    ]:
        if fraction_path is not None and not fraction_path.is_file():
            raise FileNotFoundError(f"Crop-fraction raster not found: {fraction_path}")
    if args.crop_area_fraction_band < 1:
        raise ValueError("--crop-area-fraction-band must be at least 1")
    if (
        args.crop_fraction_raster is None
        and args.crop_area_fraction_raster is None
        and args.crop_fraction_bands != "common"
    ):
        raise ValueError(
            "--crop-fraction-bands requires --crop-fraction-raster"
        )
    if (
        args.crop_area_fraction_raster is not None
        and args.crop_fraction_bands != "common"
    ):
        raise ValueError(
            "Use --crop-area-fraction-band with the legacy single-band input; "
            "--crop-fraction-bands applies to --crop-fraction-raster"
        )
    if args.start_year is not None and args.end_year is not None:
        if args.start_year > args.end_year:
            raise ValueError("--start-year must be <= --end-year")
    if args.valid_loss_min >= args.valid_loss_max:
        raise ValueError("--valid-loss-min must be < --valid-loss-max")
    if args.loss_scale <= 0:
        raise ValueError("--loss-scale must be positive")
    if args.min_valid_pixels < 1:
        raise ValueError("--min-valid-pixels must be at least 1")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")


def guard_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to "
            "replace files created by this analysis."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    rasters: Sequence[tuple[int, Path]],
    crop_match_table: pd.DataFrame,
    merged_yield: pd.DataFrame,
    map_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    plot_paths: Sequence[Path],
    crop_fraction_raster: Path | None,
    crop_fraction_band_selection: pd.DataFrame,
) -> None:
    matched = crop_match_table.loc[crop_match_table["included"]]
    common_crops = sorted(set(matched["model_crop"].astype(str)))
    source_products = sorted(set(matched["residual_product"].astype(str)))
    overlap = comparison.loc[comparison["both_sources_available"]]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "residual_csv": str(args.residual_csv.resolve()),
            "crop_config_csv": str(args.crop_config_csv.resolve()),
            "admin1_vector": str(args.admin1_vector.resolve()),
            "yield_loss_dir": str(args.yield_loss_dir.resolve()),
            "raster_pattern": args.raster_pattern,
            "crop_fraction_raster": (
                str(crop_fraction_raster.resolve())
                if crop_fraction_raster is not None
                else None
            ),
            "crop_fraction_band_specification": (
                args.crop_fraction_bands
                if args.crop_fraction_raster is not None
                else (
                    str(args.crop_area_fraction_band)
                    if args.crop_area_fraction_raster is not None
                    else None
                )
            ),
            "selected_crop_fraction_bands": (
                crop_fraction_band_selection.to_dict("records")
                if not crop_fraction_band_selection.empty
                else []
            ),
        },
        "year_selection": {
            "first_map_year": int(min(year for year, _ in rasters)),
            "last_map_year": int(max(year for year, _ in rasters)),
            "first_observed_year": int(merged_yield["year"].min()),
            "last_observed_year": int(merged_yield["year"].max()),
            "first_overlap_year": int(overlap["year"].min()) if len(overlap) else None,
            "last_overlap_year": int(overlap["year"].max()) if len(overlap) else None,
        },
        "common_model_crops": common_crops,
        "included_residual_products": source_products,
        "region_count_maps": int(map_summary["admin1_name"].nunique()),
        "region_count_observed": int(merged_yield["admin1_name"].nunique()),
        "comparison_rows_with_both_sources": int(comparison["both_sources_available"].sum()),
        "plot_count": len(plot_paths),
        "formulas": {
            "merged_observed_yield_t_ha": "sum(production_t) / sum(area_ha)",
            "merged_expected_yield_t_ha": "sum(production_trend_t) / sum(area_trend_ha)",
            "observed_signed_shortfall_fraction": "1 - observed_yield / expected_yield",
            "observed_positive_loss_fraction": "max(0, observed_signed_shortfall_fraction)",
        },
        "interpretation_notes": [
            "The modeled map is a water-stress-attributed potential yield-loss fraction.",
            "The observed detrended shortfall also contains non-water influences, data error, and crop-mix effects; correlation is validation evidence, not causal attribution.",
            "When no crop-fraction raster is supplied, Admin-1 map aggregation is pixel-area weighted over valid map cells, not weighted by the amount of crop in each pixel.",
            "With a multiband crop-fraction raster, the selected bands are summed per pixel before multiplying the pixel-area weights.",
            "Crop-fraction band selection changes only the Admin-1 spatial weights; it does not change the crop-weighted Ky already embedded in each modeled yield-loss pixel.",
            "Sorghum and Sorghum (Red) are both assigned to the model sorghum class by the default explicit aliases; use --exclude-residual-product if one series should be omitted.",
        ],
    }
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        validate_args(args)
        guard_output_directory(args.output_dir, args.overwrite)
        aliases = parse_alias_assignments(args.crop_alias)

        rasters = discover_rasters(
            args.yield_loss_dir,
            args.raster_pattern,
            args.start_year,
            args.end_year,
        )
        effective_start = args.start_year if args.start_year is not None else min(y for y, _ in rasters)
        effective_end = args.end_year if args.end_year is not None else max(y for y, _ in rasters)

        LOG.info("Preparing residual-derived yields")
        crop_level, merged_yield, crop_match_table = prepare_residual_yields(
            args.residual_csv,
            args.crop_config_csv,
            aliases,
            args.exclude_residual_product,
        )

        crop_fraction_raster = (
            args.crop_fraction_raster or args.crop_area_fraction_raster
        )
        crop_fraction_band_selection = pd.DataFrame(
            columns=[
                "band",
                "crop_name",
                "crop_name_normalized",
                "selection_source",
            ]
        )
        if crop_fraction_raster is not None:
            band_specification = (
                args.crop_fraction_bands
                if args.crop_fraction_raster is not None
                else str(args.crop_area_fraction_band)
            )
            crop_fraction_band_selection = resolve_crop_fraction_band_selection(
                crop_fraction_raster,
                band_specification,
                args.crop_config_csv,
                crop_match_table,
            )
            LOG.info(
                "Crop-fraction weighting uses band(s) %s: %s",
                ",".join(
                    crop_fraction_band_selection["band"].astype(str).tolist()
                ),
                ", ".join(
                    crop_fraction_band_selection["crop_name"].astype(str).tolist()
                ),
            )

        crop_level = crop_level.loc[
            crop_level["year"].between(effective_start, effective_end)
        ].copy()
        merged_yield = merged_yield.loc[
            merged_yield["year"].between(effective_start, effective_end)
        ].copy()
        if merged_yield.empty:
            raise ValueError(
                "Residual dataset has no common-crop yield records within the "
                f"selected map period {effective_start}-{effective_end}"
            )

        LOG.info("Reading and matching Admin-1 regions")
        admin = load_admin1(args.admin1_vector, args.admin_layer, args.admin_name_field)
        crop_level, merged_yield, region_crosswalk = map_residual_regions(
            crop_level,
            merged_yield,
            admin,
            args.admin_name_field,
            args.allow_unmatched_regions,
        )

        LOG.info("Aggregating annual yield-loss maps to Admin-1")
        map_summary = aggregate_yield_loss_rasters(
            rasters=rasters,
            admin=admin,
            admin_name_field=args.admin_name_field,
            raster_band=args.raster_band,
            spatial_weighting=args.spatial_weighting,
            crop_fraction_raster=crop_fraction_raster,
            crop_fraction_bands=(
                crop_fraction_band_selection["band"].astype(int).tolist()
                if not crop_fraction_band_selection.empty
                else []
            ),
            all_touched=args.all_touched,
            loss_scale=args.loss_scale,
            valid_loss_min=args.valid_loss_min,
            valid_loss_max=args.valid_loss_max,
            min_valid_pixels=args.min_valid_pixels,
        )

        comparison, statistics = build_comparison(map_summary, merged_yield)

        output_dir = args.output_dir
        crop_match_table.to_csv(output_dir / "crop_match_table.csv", index=False)
        crop_fraction_band_selection.to_csv(
            output_dir / "crop_fraction_band_selection.csv", index=False
        )
        region_crosswalk.to_csv(output_dir / "region_match_table.csv", index=False)
        crop_level.to_csv(output_dir / "crop_yield_from_residuals_admin1.csv", index=False)
        merged_yield.to_csv(
            output_dir / "merged_common_crop_yield_from_residuals_admin1.csv", index=False
        )
        map_summary.to_csv(output_dir / "modeled_yield_loss_admin1.csv", index=False)
        comparison.to_csv(output_dir / "yield_loss_yield_comparison_admin1.csv", index=False)
        statistics.to_csv(output_dir / "yield_loss_yield_statistics_admin1.csv", index=False)

        plot_paths: list[Path] = []
        if not args.no_plots:
            LOG.info("Creating one comparison plot per Admin-1 region")
            plot_paths = plot_admin1_comparisons(
                comparison,
                statistics,
                output_dir / "plots",
                args.dpi,
            )

        write_metadata(
            output_dir / "run_metadata.json",
            args,
            rasters,
            crop_match_table,
            merged_yield,
            map_summary,
            comparison,
            plot_paths,
            crop_fraction_raster,
            crop_fraction_band_selection,
        )

        common_crops = sorted(
            set(crop_match_table.loc[crop_match_table["included"], "model_crop"].astype(str))
        )
        overlap = comparison.loc[comparison["both_sources_available"]]
        LOG.info("Common modeled crops: %s", ", ".join(common_crops))
        LOG.info(
            "Finished: %d Admin-1 regions, %d maps, %d overlapping region-years, %d plots",
            map_summary["admin1_name"].nunique(),
            len(rasters),
            len(overlap),
            len(plot_paths),
        )
        LOG.info("Outputs: %s", output_dir.resolve())
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
