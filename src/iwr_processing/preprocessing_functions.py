"""
Processing functions for the IWR workflow.

Each public function in this module can be referenced by name in the
WORKFLOW.PROCESS[*].process_list[*].function config key.
"""

import logging
import os
import tempfile
import csv
import re

logger = logging.getLogger(__name__)


def _normalize_crop_token(value: str) -> str:
    token = str(value).strip().lower()
    token = token.replace("-", "_").replace("/", "_")
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def _parse_first_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return None
    return float(match.group(0))


def _as_class_list(classes: list[str | int] | str | None) -> list[str | int]:
    if classes is None:
        return []
    if isinstance(classes, list):
        return classes
    return [item.strip() for item in classes.split(",") if item.strip()]


def _load_class_map(class_map_csv: str) -> dict[str, int]:
    """Load a name->value class map from a CSV file."""
    with open(class_map_csv, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        headers = [h or "" for h in (reader.fieldnames or [])]
        lower = {h.lower(): h for h in headers}

        value_col = (
            lower.get("value")
            or lower.get("code")
            or lower.get("id")
            or lower.get("class_value")
            or lower.get("raster_value")
        )
        label_col = (
            lower.get("land_cover")
            or lower.get("label")
            or lower.get("name")
            or lower.get("class_name")
            or lower.get("land_cover_name")
        )

        if not value_col or not label_col:
            raise ValueError(
                "CSV class map must contain value/code and label/name columns"
            )

        out: dict[str, int] = {}
        for row in reader:
            raw_value = str(row[value_col]).strip()
            raw_label = str(row[label_col]).strip()
            if not raw_value or not raw_label:
                continue
            out[raw_label.lower()] = int(float(raw_value))

    if not out:
        raise ValueError(f"No classes found in CSV map: {class_map_csv}")
    return out


def match_grid(
    inputs: list[str],
    grid: str,
    nodata_values: list[int | float] | None = None,
    resampling_method: str = "nearest",
    output: str | None = None,
    **kwargs,
) -> list[str]:
    """Reproject/resample a list of rasters to match a reference grid.

    Parameters
    ----------
    inputs:
        Raster file paths to align.
    grid:
        Path to the reference grid raster that defines the target CRS,
        extent, and resolution.
    nodata_values:
        Per-input nodata values (positional, same order as *inputs*).
        Values matching these will be masked before reprojection.
    resampling_method:
        Rasterio resampling algorithm name, e.g. 'nearest', 'bilinear',
        'mode'. Case-insensitive.
    output:
        When provided *and* only one input is given (i.e. the function is
        the last step in a chain), the aligned raster is written directly
        to this path instead of a temporary file.

    Returns
    -------
    list[str]
        Paths of the aligned raster files (temp files or ``[output]``).
    """
    import rioxarray as rxr
    from rasterio.enums import Resampling

    try:
        resampling_enum = Resampling[resampling_method.lower()]
    except KeyError:
        logger.warning(
            "Unknown resampling method '%s', falling back to 'nearest'.",
            resampling_method,
        )
        resampling_enum = Resampling.nearest

    reference = rxr.open_rasterio(grid, masked=True)
    logger.info("Reference grid  CRS   : %s", reference.rio.crs)
    logger.info("Reference grid  shape : %s", reference.rio.shape)

    aligned_paths: list[str] = []

    for i, inp in enumerate(inputs):
        nodata = nodata_values[i] if nodata_values and i < len(nodata_values) else None

        ds = rxr.open_rasterio(inp, masked=True)
        if nodata is not None:
            ds = ds.where(ds != nodata)

        aligned = ds.rio.reproject_match(reference, resampling=resampling_enum)

        # Write to the final output path when single input and output given,
        # otherwise write to a temporary file so merge_rasters can consume it.
        if output is not None and len(inputs) == 1:
            out_path = output
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".tif", prefix="iwr_matched_", delete=False
            )
            tmp.close()
            out_path = tmp.name

        aligned.rio.to_raster(out_path)
        logger.info("  [%d/%d] aligned: %s -> %s", i + 1, len(inputs), inp, out_path)
        aligned_paths.append(out_path)

    return aligned_paths


def merge_rasters(
    inputs: list[str],
    output: str | None = None,
    output_nodata: int | float = 255,
    **kwargs,
) -> str:
    """Merge a list of already-aligned rasters into a single output file.

    Inputs are expected to share the same grid (e.g. produced by
    :func:`match_grid`).  Where pixels overlap the first raster in the
    list takes priority (rasterio default).

    Parameters
    ----------
    inputs:
        Aligned raster file paths to merge.
    output:
        Destination path for the merged raster. If omitted, a temporary
        GeoTIFF is created and returned so this function can be used in the
        middle of a process chain.
    output_nodata:
        Nodata value to write in the output.

    Returns
    -------
    str
        Path of the written output file.
    """
    import rasterio
    from rasterio.merge import merge

    datasets = [rasterio.open(f) for f in inputs]

    mosaic, transform = merge(datasets, nodata=output_nodata)

    meta = datasets[0].meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "nodata": output_nodata,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )

    for ds in datasets:
        ds.close()

    out_path = output or tempfile.NamedTemporaryFile(
        suffix=".tif", prefix="iwr_merged_", delete=False
    ).name

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dest:
        dest.write(mosaic)

    logger.info("Merged %d raster(s) -> %s", len(inputs), out_path)

    # Remove temp files produced by match_grid.
    tmp_dir = tempfile.gettempdir()
    for f in inputs:
        if os.path.abspath(f).startswith(tmp_dir):
            try:
                os.unlink(f)
            except OSError:
                pass

    return out_path


def aggregate_fractional_layers(
    grid: str,
    class_values: list[int | float] | str | None = None,
    classes: list[str | int] | str | None = None,
    class_map_csv: str | None = None,
    land_cover: str | None = None,
    input: str | None = None,
    inputs: list[str] | None = None,
    output: str | None = None,
    output_nodata: float = -9999.0,
    source_nodata: int | float | None = None,
    **kwargs,
) -> str:
    """Create a multiband raster with class fractions (%) per target-grid cell.

    Preferred usage is to pass *class_values* as numeric land-cover codes.
    Name-based *classes* is also supported for backward compatibility and
    requires *class_map_csv*.

    Each output band corresponds to one requested class value.
    Pixel values are percentages in [0, 100], computed from a higher
    resolution categorical land-cover raster aggregated onto *grid*.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    def _normalize_band_description(label: str) -> str:
        normalized = str(label).strip().lower()
        normalized = normalized.replace("/", "_").replace("-", "_")
        normalized = "_".join(part for part in normalized.split() if part)
        return normalized

    source_path = land_cover or input or (inputs[0] if inputs else None)
    if source_path is None:
        raise ValueError(
            "Missing input land-cover raster. Provide 'land_cover' or 'input'."
        )

    class_map = _load_class_map(class_map_csv) if class_map_csv else {}

    class_values_list = _as_class_list(class_values)
    class_name_list = _as_class_list(classes)

    if not class_values_list and not class_name_list:
        raise ValueError("Provide either 'class_values' or 'classes'.")

    class_codes: list[int] = []
    class_labels: list[str] = []
    if class_values_list:
        reverse_map = {v: k for k, v in class_map.items()} if class_map else {}
        for idx, c in enumerate(class_values_list):
            code = int(float(c))
            class_codes.append(code)
            if class_name_list and idx < len(class_name_list):
                class_labels.append(str(class_name_list[idx]))
            else:
                class_labels.append(reverse_map.get(code, str(code)))
    else:
        if not class_map_csv:
            raise ValueError("'classes' requires 'class_map_csv' to resolve names.")
        for c in class_name_list:
            token = str(c).strip()
            key = token.lower()
            if key not in class_map:
                raise ValueError(f"Class '{token}' not found in map '{class_map_csv}'")
            class_codes.append(class_map[key])
            class_labels.append(token)

    with rasterio.open(source_path) as src, rasterio.open(grid) as grd:
        src_arr = src.read(1)
        src_nodata = src.nodata if source_nodata is None else source_nodata

        dst_shape = (grd.height, grd.width)
        dst_transform = grd.transform
        dst_crs = grd.crs

        if src_nodata is None:
            valid_mask = np.ones(src_arr.shape, dtype=np.float32)
        else:
            valid_mask = (src_arr != src_nodata).astype(np.float32)

        valid_fraction = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=valid_mask,
            destination=valid_fraction,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
        )

        out_stack = np.full(
            (len(class_codes), grd.height, grd.width),
            output_nodata,
            dtype=np.float32,
        )

        for idx, code in enumerate(class_codes):
            indicator = ((src_arr == code) & (valid_mask > 0)).astype(np.float32)

            class_fraction = np.zeros(dst_shape, dtype=np.float32)
            reproject(
                source=indicator,
                destination=class_fraction,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                percent = np.where(
                    valid_fraction > 0,
                    (class_fraction / valid_fraction) * 100.0,
                    output_nodata,
                )
            clipped = np.where(
                valid_fraction > 0,
                np.clip(percent, 0.0, 100.0),
                output_nodata,
            )
            out_stack[idx, :, :] = clipped.astype(np.float32)

        out_path = output or tempfile.NamedTemporaryFile(
            suffix=".tif", prefix="iwr_fractional_", delete=False
        ).name

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        profile = grd.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "count": len(class_codes),
                "dtype": "float32",
                "nodata": output_nodata,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_stack)
            for bidx, label in enumerate(class_labels, start=1):
                dst.set_band_description(bidx, f"frac_{_normalize_band_description(label)}")

    logger.info(
        "Fractional layers written: %s (bands=%d)", out_path, len(class_codes)
    )
    return out_path


def apply_raster_mask(
    mask: str,
    filter_values: list[float | int] | str | None = None,
    nodata_value: float | int | None = None,
    input: str | None = None,
    inputs: list[str] | str | None = None,
    output: str | None = None,
    **kwargs,
) -> str | list[str]:
    """Mask raster(s) where mask values match a filter list.

    This is the file-path adaptation of the DataArray utility. It supports
    your workflow style, including calls like:

    {"function": "apply_raster_mask", "mask": "{DATASETS.grid}"}

    In that form, the input comes from the previous step through the runner
    pipe (as ``inputs``), and the step output is injected automatically.
    """
    import numpy as np
    import rioxarray as rxr

    # Normalize input paths from either input or inputs.
    path_list: list[str] = []
    if input:
        path_list.append(input)
    if inputs is not None:
        if isinstance(inputs, str):
            path_list.append(inputs)
        else:
            path_list.extend(inputs)

    if not path_list:
        raise ValueError("Missing input raster. Provide 'input' or 'inputs'.")

    # Normalize filter values. If set to 'none', this function is passthrough.
    if isinstance(filter_values, str):
        if filter_values.lower() == "none":
            filter_list: list[float | int] = []
            passthrough = True
        else:
            filter_list = [
                float(v.strip())
                for v in filter_values.split(",")
                if v.strip()
            ]
            passthrough = False
    elif filter_values is None:
        filter_list = []
        passthrough = False
    else:
        filter_list = list(filter_values)
        passthrough = False

    mask_da = rxr.open_rasterio(mask, masked=True)
    if "band" in mask_da.dims and mask_da.sizes.get("band", 1) == 1:
        mask_da = mask_da.squeeze("band", drop=True)

    out_paths: list[str] = []

    for idx, in_path in enumerate(path_list):
        data = rxr.open_rasterio(in_path, masked=True)

        # Use input nodata when possible, otherwise fallback to -9999.
        effective_nodata = nodata_value
        if effective_nodata is None:
            effective_nodata = data.rio.nodata
        if effective_nodata is None:
            effective_nodata = data.attrs.get("_FillValue")
        if effective_nodata is None:
            effective_nodata = -9999.0

        # Reproject mask to data grid before applying value-based filter.
        mask_matched = mask_da.rio.reproject_match(data)

        if passthrough:
            masked_data = data.rio.write_nodata(effective_nodata)
        else:
            mask_nodata = mask_matched.rio.nodata
            values_to_mask = list(filter_list)
            if mask_nodata is not None:
                values_to_mask.append(mask_nodata)

            masked_data = data
            for value in values_to_mask:
                masked_data = masked_data.where(
                    ~np.isclose(mask_matched, value, equal_nan=True),
                    other=effective_nodata,
                )
            masked_data = masked_data.rio.write_nodata(effective_nodata)

        # If single input and output provided, write there; otherwise temp files.
        if output is not None and len(path_list) == 1:
            out_path = output
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".tif", prefix="iwr_masked_", delete=False
            )
            tmp.close()
            out_path = tmp.name

        masked_data.rio.to_raster(out_path)
        out_paths.append(out_path)
        logger.info("  [%d/%d] masked: %s -> %s", idx + 1, len(path_list), in_path, out_path)

    if len(out_paths) == 1:
        return out_paths[0]
    return out_paths


def build_crop_calendar_template_from_raster(
    land_cover_raster: str,
    fao56_kc_csv: str,
    output: str | None = None,
    class_map_csv: str | None = None,
    min_fraction_threshold: float = 0.0,
    kc_off_season: float = 0.5,
    **kwargs,
) -> str:
    """Build a crop-calendar template CSV from crop raster coverage.

    The output template intentionally has no ``year`` column and leaves
    ``planting_doy`` / ``harvest_doy`` empty for manual completion.

    Input raster can be either:
    - multiband fractional raster with crop band descriptions (preferred), or
    - single-band categorical raster using ``class_map_csv`` to map values to crop IDs.
    """
    import numpy as np
    import pandas as pd
    import rasterio

    if output is None:
        raise ValueError("'output' path is required.")
    if not os.path.exists(land_cover_raster):
        raise FileNotFoundError(f"Land-cover raster not found: {land_cover_raster}")
    if not os.path.exists(fao56_kc_csv):
        raise FileNotFoundError(f"FAO56 KC CSV not found: {fao56_kc_csv}")

    class_name_by_code: dict[int, str] = {}
    if class_map_csv:
        with open(class_map_csv, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            headers = [h or "" for h in (reader.fieldnames or [])]
            lower = {h.lower(): h for h in headers}

            value_col = (
                lower.get("value")
                or lower.get("code")
                or lower.get("id")
                or lower.get("class_value")
                or lower.get("raster_value")
            )
            label_col = (
                lower.get("crop_id")
                or lower.get("land_cover")
                or lower.get("label")
                or lower.get("name")
                or lower.get("class_name")
                or lower.get("land_cover_name")
            )

            if not value_col or not label_col:
                raise ValueError(
                    "class_map_csv must contain value/code and crop/name columns"
                )

            for row in reader:
                raw_value = str(row[value_col]).strip()
                raw_label = str(row[label_col]).strip()
                if not raw_value or not raw_label:
                    continue
                class_name_by_code[int(float(raw_value))] = raw_label

    used_crop_tokens: list[str] = []
    with rasterio.open(land_cover_raster) as src:
        # Multiband fractional raster: use band descriptions and keep bands with > threshold.
        if src.count > 1:
            nodata = src.nodata
            for band_idx in range(1, src.count + 1):
                arr = src.read(band_idx).astype(np.float32)
                if nodata is not None:
                    arr = np.where(np.isclose(arr, nodata), np.nan, arr)
                max_val = np.nanmax(arr) if arr.size else np.nan
                if np.isnan(max_val) or max_val <= float(min_fraction_threshold):
                    continue

                desc = src.descriptions[band_idx - 1] or f"band_{band_idx}"
                token = _normalize_crop_token(desc)
                if token.startswith("frac_"):
                    token = token[5:]
                used_crop_tokens.append(token)
        else:
            # Single-band categorical raster: use mapped class names if available.
            arr = src.read(1)
            nodata = src.nodata
            if nodata is None:
                unique_codes = np.unique(arr)
            else:
                unique_codes = np.unique(arr[~np.isclose(arr, nodata)])

            for code in unique_codes:
                int_code = int(float(code))
                label = class_name_by_code.get(int_code, str(int_code))
                used_crop_tokens.append(_normalize_crop_token(label))

    used_crop_tokens = sorted(set(token for token in used_crop_tokens if token))
    if not used_crop_tokens:
        raise ValueError("No crop classes detected in input raster.")

    fao_df = pd.read_csv(fao56_kc_csv)
    required_cols = {"crop_id", "crop_name_fao56", "kc_ini", "kc_mid", "kc_end"}
    missing_cols = required_cols - set(fao_df.columns)
    if missing_cols:
        raise ValueError(
            f"FAO56 KC CSV is missing required columns: {sorted(missing_cols)}"
        )

    fao_df["crop_id_norm"] = fao_df["crop_id"].map(_normalize_crop_token)
    fao_df["crop_name_norm"] = fao_df["crop_name_fao56"].map(_normalize_crop_token)

    aliases = {
        "wheat": "winter_wheat",
        "maize": "maize_field_grain_field_corn",
        "corn": "maize_field_grain_field_corn",
    }

    out_rows: list[dict[str, object]] = []
    for token in used_crop_tokens:
        token_resolved = aliases.get(token, token)

        row = fao_df.loc[fao_df["crop_id_norm"] == token_resolved]
        if row.empty:
            row = fao_df.loc[fao_df["crop_name_norm"] == token_resolved]
        if row.empty:
            # Fallback: substring match to help with close naming variants.
            row = fao_df.loc[
                fao_df["crop_id_norm"].str.contains(token_resolved, regex=False)
                | fao_df["crop_name_norm"].str.contains(token_resolved, regex=False)
            ]

        if row.empty:
            logger.warning("No FAO56 KC match for crop token '%s'", token)
            crop_id = token
            kc_ini = ""
            kc_mid = ""
            kc_end = ""
        else:
            chosen = row.iloc[0]
            crop_id = str(chosen["crop_id"])
            kc_ini_val = _parse_first_float(chosen.get("kc_ini"))
            kc_mid_val = _parse_first_float(chosen.get("kc_mid"))
            kc_end_val = _parse_first_float(chosen.get("kc_end"))
            kc_ini = "" if kc_ini_val is None else round(kc_ini_val, 3)
            kc_mid = "" if kc_mid_val is None else round(kc_mid_val, 3)
            kc_end = "" if kc_end_val is None else round(kc_end_val, 3)

        out_rows.append(
            {
                "crop_id": crop_id,
                "planting_doy": "",
                "harvest_doy": "",
                "kc_ini": kc_ini,
                "kc_mid": kc_mid,
                "kc_end": kc_end,
                "growth_stage_days_initial": "",
                "growth_stage_days_development": "",
                "growth_stage_days_mid_season": "",
                "growth_stage_days_late_season": "",
                "kc_off_season": kc_off_season,
            }
        )

    output_df = pd.DataFrame(out_rows)
    output_df = output_df.sort_values("crop_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    output_df.to_csv(output, index=False)

    logger.info(
        "Crop calendar template written: %s (crops=%d)", output, len(output_df)
    )
    return output
