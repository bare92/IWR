from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS


USDA_TEXTURE: dict[int, dict[str, float | str]] = {
    1: {"name": "Clay heavy", "sand": 20.0, "silt": 20.0, "clay": 60.0},
    2: {"name": "Silty clay", "sand": 10.0, "silt": 45.0, "clay": 45.0},
    3: {"name": "Clay", "sand": 30.0, "silt": 20.0, "clay": 50.0},
    4: {"name": "Silty clay loam", "sand": 10.0, "silt": 55.0, "clay": 35.0},
    5: {"name": "Clay loam", "sand": 32.0, "silt": 34.0, "clay": 34.0},
    6: {"name": "Silt", "sand": 8.0, "silt": 88.0, "clay": 4.0},
    7: {"name": "Silt loam", "sand": 20.0, "silt": 65.0, "clay": 15.0},
    8: {"name": "Sandy clay", "sand": 55.0, "silt": 10.0, "clay": 35.0},
    9: {"name": "Loam", "sand": 40.0, "silt": 40.0, "clay": 20.0},
    10: {"name": "Sandy clay loam", "sand": 60.0, "silt": 15.0, "clay": 25.0},
    11: {"name": "Sandy loam", "sand": 65.0, "silt": 25.0, "clay": 10.0},
    12: {"name": "Loamy sand", "sand": 80.0, "silt": 15.0, "clay": 5.0},
    13: {"name": "Sand", "sand": 92.0, "silt": 5.0, "clay": 3.0},
}

USDA_DEFAULT_OM: dict[int, float] = {
    1: 2.5,
    2: 2.2,
    3: 2.0,
    4: 2.0,
    5: 1.8,
    6: 2.0,
    7: 1.8,
    8: 1.4,
    9: 1.6,
    10: 1.3,
    11: 1.2,
    12: 0.8,
    13: 0.5,
}


def aquacrop_hydraulic_from_texture(
    sand_pct: float,
    clay_pct: float,
    org_mat: float,
    df: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute AquaCrop-like hydraulic properties from texture percentages."""
    if sand_pct < 0 or clay_pct < 0:
        raise ValueError("sand_pct and clay_pct must be >= 0")
    if sand_pct + clay_pct > 100:
        raise ValueError("sand_pct + clay_pct cannot exceed 100")
    if df <= 0:
        raise ValueError("df must be > 0")

    sand = sand_pct / 100.0
    clay = clay_pct / 100.0

    pred_th_wp = (
        -(0.024 * sand)
        + (0.487 * clay)
        + (0.006 * org_mat)
        + (0.005 * sand * org_mat)
        - (0.013 * clay * org_mat)
        + (0.068 * sand * clay)
        + 0.031
    )
    th_wp = pred_th_wp + (0.14 * pred_th_wp) - 0.02

    pred_th_fc = (
        -(0.251 * sand)
        + (0.195 * clay)
        + (0.011 * org_mat)
        + (0.006 * sand * org_mat)
        - (0.027 * clay * org_mat)
        + (0.452 * sand * clay)
        + 0.299
    )
    pred_adj_th_fc = pred_th_fc + ((1.283 * (pred_th_fc ** 2)) - (0.374 * pred_th_fc) - 0.015)

    pred_th_s33 = (
        (0.278 * sand)
        + (0.034 * clay)
        + (0.022 * org_mat)
        - (0.018 * sand * org_mat)
        - (0.027 * clay * org_mat)
        - (0.584 * sand * clay)
        + 0.078
    )
    pred_adj_th_s33 = pred_th_s33 + ((0.636 * pred_th_s33) - 0.107)
    pred_th_s = (pred_adj_th_fc + pred_adj_th_s33) + ((-0.097 * sand) + 0.043)

    p_n = (1 - pred_th_s) * 2.65
    p_df = p_n * df
    poros_comp = (1 - (p_df / 2.65)) - (1 - (p_n / 2.65))
    poros_comp_om = 1 - (p_df / 2.65)

    th_fc = pred_adj_th_fc + (0.2 * poros_comp)
    th_s = poros_comp_om

    if th_fc <= 0 or th_wp <= 0 or abs(th_fc - th_wp) < 1e-12:
        ksat = float("nan")
    else:
        lmbda = 1.0 / ((math.log(1500.0) - math.log(33.0)) / (math.log(th_fc) - math.log(th_wp)))
        ksat = (1930.0 * ((th_s - th_fc) ** (3.0 - lmbda))) * 24.0

    return (
        round(th_wp, 3),
        round(th_fc, 3),
        round(th_s, 3),
        round(ksat, 1) if not math.isnan(ksat) else ksat,
    )


def class_to_fc_wp(
    class_id: int,
    om_override: float | None = None,
    nodata_value: float | None = None,
) -> tuple[float | None, float | None]:
    """Convert one USDA class id to (fc, wp)."""
    if class_id == 0 or class_id not in USDA_TEXTURE:
        return nodata_value, nodata_value

    tex = USDA_TEXTURE[class_id]
    om = USDA_DEFAULT_OM[class_id] if om_override is None else om_override
    wp, fc, _ths, _ksat = aquacrop_hydraulic_from_texture(
        sand_pct=float(tex["sand"]),
        clay_pct=float(tex["clay"]),
        org_mat=om,
        df=1.0,
    )
    return fc, wp


def _normalize_crop_name(name: str) -> str:
    """Normalize crop names before matching across rasters and tables."""
    normalized = name.strip().lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def find_most_similar_name(
    raster_name: str,
    candidate_names: Sequence[str],
) -> tuple[str | None, float]:
    """Return the closest candidate name and similarity score in [0, 1]."""
    if not candidate_names:
        return None, 0.0

    normalized_raster = _normalize_crop_name(raster_name)
    best_name: str | None = None
    best_score = -1.0

    for candidate in candidate_names:
        normalized_candidate = _normalize_crop_name(candidate)

        if normalized_candidate == normalized_raster:
            return candidate, 1.0

        score = SequenceMatcher(None, normalized_raster, normalized_candidate).ratio()
        if score > best_score:
            best_name = candidate
            best_score = score

    return best_name, best_score


@dataclass
class CropParameter:
    crop_id: str
    crop_name_fao56: str
    category: str
    root_depth_min_m: float
    root_depth_max_m: float
    p_table22_for_ETc_5mm_day: float
    source: str | None = None
    notes: str | None = None

    def p_adjusted(self, etc_mm_day: float) -> float:
        """FAO-56 adjustment of p for ETc different from 5 mm/day."""
        p = self.p_table22_for_ETc_5mm_day + 0.04 * (5.0 - etc_mm_day)
        return float(np.clip(p, 0.1, 0.8))


@dataclass
class CropFraction:
    crop_type: list[str]
    raster: np.ndarray  # shape: (bands, rows, cols)
    nodata: float | int | None
    transform: Affine
    crs: CRS | None
    path: str

    # New optional field
    crop_parameters: dict[str, CropParameter] | None = None

    @classmethod
    def from_geotiff(
        cls,
        geotiff_path: str,
        crop_type: Sequence[str] | None = None,
        crop_parameter_csv: str | None = None,
    ) -> "CropFraction":
        """Load a multiband fractional GeoTIFF into CropFraction.

        If crop_type is not provided, crop names are inferred from band descriptions.
        Example description: 'frac_wheat' -> crop type 'wheat'.

        If crop_parameter_csv is provided, FAO-56 crop parameters are loaded and
        matched using the crop_id column.
        """
        p = Path(geotiff_path)
        if not p.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

        with rasterio.open(p) as src:
            raster = src.read().astype(np.float32)
            band_count = src.count
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

            desc = list(src.descriptions or [])
            desc = [d if d else f"band_{i+1}" for i, d in enumerate(desc)]

        inferred_crop_types = [
            d.replace("frac_", "", 1) if d.startswith("frac_") else d
            for d in desc
        ]

        if crop_type is None:
            crop_list = inferred_crop_types
        else:
            crop_list = [str(c) for c in crop_type]
            if len(crop_list) != band_count:
                raise ValueError(
                    f"crop_type length ({len(crop_list)}) must match number "
                    f"of bands ({band_count})."
                )

        crop_parameters = None
        if crop_parameter_csv is not None:
            crop_parameters = load_crop_parameters(crop_parameter_csv)

            missing = [c for c in crop_list if c not in crop_parameters]
            if missing:
                suggestions: list[str] = []
                for missing_name in missing:
                    best_name, best_score = find_most_similar_name(
                        missing_name,
                        list(crop_parameters.keys()),
                    )
                    if best_name is None:
                        suggestions.append(f"{missing_name!r} -> no suggestion")
                    else:
                        suggestions.append(
                            f"{missing_name!r} -> {best_name!r} (score={best_score:.2f})"
                        )

                raise ValueError(
                    "Some raster crop names are missing from the crop parameter table: "
                    f"{missing}. Closest matches: {suggestions}. "
                    "Check that raster band names match the crop_id column."
                )

        return cls(
            crop_type=crop_list,
            raster=raster,
            nodata=nodata,
            transform=transform,
            crs=crs,
            path=str(p),
            crop_parameters=crop_parameters,
        )

    def get_crop_fraction(self, crop: str) -> np.ndarray:
        """Return the 2D fraction raster for a given crop name."""
        if crop not in self.crop_type:
            raise KeyError(f"Crop '{crop}' not found. Available: {self.crop_type}")
        idx = self.crop_type.index(crop)
        return self.raster[idx]

    def find_similar_crop_name_in_table(self, crop: str) -> tuple[str | None, float]:
        """Return the closest crop name available in the loaded parameter table."""
        if self.crop_parameters is None:
            raise ValueError("No crop parameter table was loaded.")

        return find_most_similar_name(crop, list(self.crop_parameters.keys()))

    def get_crop_parameter(self, crop: str) -> CropParameter:
        """Return FAO-56 parameters for a given crop."""
        if self.crop_parameters is None:
            raise ValueError("No crop parameter table was loaded.")

        if crop not in self.crop_parameters:
            best_name, best_score = self.find_similar_crop_name_in_table(crop)
            if best_name is None:
                raise KeyError(
                    f"Crop '{crop}' not found in parameter table. "
                    f"Available: {list(self.crop_parameters)}"
                )

            raise KeyError(
                f"Crop '{crop}' not found in parameter table. "
                f"Closest match is '{best_name}' (score={best_score:.2f})."
            )

        return self.crop_parameters[crop]

    def get_crop_root_depth(self, crop: str, use: str = "max") -> float:
        """Return crop rooting depth.

        use='min', 'max', or 'mean'
        """
        param = self.get_crop_parameter(crop)

        if use == "min":
            return param.root_depth_min_m
        if use == "max":
            return param.root_depth_max_m
        if use == "mean":
            return 0.5 * (param.root_depth_min_m + param.root_depth_max_m)

        raise ValueError("use must be one of: 'min', 'max', 'mean'")

    def get_crop_p(self, crop: str, etc_mm_day: float | None = None) -> float:
        """Return FAO-56 depletion fraction p.

        If etc_mm_day is provided, p is adjusted using the FAO-56 correction.
        Otherwise, the Table 22 value for ETc = 5 mm/day is returned.
        """
        param = self.get_crop_parameter(crop)

        if etc_mm_day is None:
            return param.p_table22_for_ETc_5mm_day

        return param.p_adjusted(etc_mm_day)


def load_crop_parameters(csv_path: str) -> dict[str, CropParameter]:
    """Load FAO-56 crop parameters from CSV.

    Expected columns:
    - crop_id
    - crop_name_fao56
    - category
    - root_depth_min_m
    - root_depth_max_m
    - p_table22_for_ETc_5mm_day
    """
    df = pd.read_csv(csv_path)

    required_columns = {
        "crop_id",
        "crop_name_fao56",
        "category",
        "root_depth_min_m",
        "root_depth_max_m",
        "p_table22_for_ETc_5mm_day",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Crop parameter table is missing columns: {sorted(missing_columns)}"
        )

    crop_parameters: dict[str, CropParameter] = {}

    for _, row in df.iterrows():
        crop_id = str(row["crop_id"])

        crop_parameters[crop_id] = CropParameter(
            crop_id=crop_id,
            crop_name_fao56=str(row["crop_name_fao56"]),
            category=str(row["category"]),
            root_depth_min_m=float(row["root_depth_min_m"]),
            root_depth_max_m=float(row["root_depth_max_m"]),
            p_table22_for_ETc_5mm_day=float(row["p_table22_for_ETc_5mm_day"]),
            source=str(row["source"]) if "source" in df.columns and pd.notna(row["source"]) else None,
            notes=str(row["notes"]) if "notes" in df.columns and pd.notna(row["notes"]) else None,
        )

    return crop_parameters


@dataclass
class SoilLayer:
    soil_type: np.ndarray  # shape: (rows, cols), USDA class ids
    field_capacity: np.ndarray  # shape: (rows, cols), m3/m3
    wilting_point: np.ndarray  # shape: (rows, cols), m3/m3
    nodata: float | int | None
    hydraulic_nodata: float
    transform: Affine
    crs: CRS | None
    path: str
    taw_mm: np.ndarray | None = None  # shape: (rows, cols), mm; WATNEEDS direct input
    smax_mm: np.ndarray | None = None  # shape: (rows, cols), mm; max soil storage

    @classmethod
    def from_geotiff(
        cls,
        geotiff_path: str,
        field_capacity: np.ndarray | None = None,
        wilting_point: np.ndarray | None = None,
        taw_mm: np.ndarray | str | None = None,
        smax_mm: np.ndarray | str | None = None,
        om_override: float | None = None,
        output_nodata: float = -9999.0,
    ) -> "SoilLayer":
        """Load soil classes and derive FC/WP if they are not provided.
        
        If taw_mm or smax_mm are provided as raster paths (str), they are loaded.
        Otherwise, FC/WP are derived from USDA texture via pedotransfer (default).
        """
        p = Path(geotiff_path)
        if not p.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

        with rasterio.open(p) as src:
            arr = src.read(1)
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

        if field_capacity is None or wilting_point is None:
            fc_arr = np.full(arr.shape, output_nodata, dtype=np.float32)
            wp_arr = np.full(arr.shape, output_nodata, dtype=np.float32)

            # Compute once per unique texture class for speed and clarity.
            unique_classes = np.unique(arr)
            for class_val in unique_classes:
                if nodata is not None and np.isclose(class_val, nodata):
                    continue

                try:
                    class_id = int(class_val)
                except (TypeError, ValueError):
                    continue

                fc, wp = class_to_fc_wp(
                    class_id=class_id,
                    om_override=om_override,
                    nodata_value=output_nodata,
                )
                if fc is None or wp is None:
                    continue

                mask = np.isclose(arr, class_val)
                fc_arr[mask] = fc
                wp_arr[mask] = wp
        else:
            fc_arr = field_capacity.astype(np.float32)
            wp_arr = wilting_point.astype(np.float32)

        if fc_arr.shape != arr.shape or wp_arr.shape != arr.shape:
            raise ValueError("field_capacity and wilting_point must match soil raster shape")

        # Load optional WATNEEDS-direct inputs
        taw_mm_arr = None
        smax_mm_arr = None
        
        if taw_mm is not None:
            if isinstance(taw_mm, str):
                with rasterio.open(taw_mm) as src:
                    taw_mm_arr = src.read(1).astype(np.float32)
                    if taw_mm_arr.shape != arr.shape:
                        raise ValueError(
                            f"taw_mm raster shape {taw_mm_arr.shape} does not match "
                            f"soil raster shape {arr.shape}"
                        )
            else:
                taw_mm_arr = np.asarray(taw_mm, dtype=np.float32)
                if taw_mm_arr.shape != arr.shape:
                    raise ValueError(
                        f"taw_mm array shape {taw_mm_arr.shape} does not match "
                        f"soil raster shape {arr.shape}"
                    )
        
        if smax_mm is not None:
            if isinstance(smax_mm, str):
                with rasterio.open(smax_mm) as src:
                    smax_mm_arr = src.read(1).astype(np.float32)
                    if smax_mm_arr.shape != arr.shape:
                        raise ValueError(
                            f"smax_mm raster shape {smax_mm_arr.shape} does not match "
                            f"soil raster shape {arr.shape}"
                        )
            else:
                smax_mm_arr = np.asarray(smax_mm, dtype=np.float32)
                if smax_mm_arr.shape != arr.shape:
                    raise ValueError(
                        f"smax_mm array shape {smax_mm_arr.shape} does not match "
                        f"soil raster shape {arr.shape}"
                    )
        
        return cls(
            soil_type=arr,
            field_capacity=fc_arr,
            wilting_point=wp_arr,
            nodata=nodata,
            hydraulic_nodata=output_nodata,
            transform=transform,
            crs=crs,
            path=str(p),
            taw_mm=taw_mm_arr,
            smax_mm=smax_mm_arr,
        )

    def hydraulic_mask(self) -> np.ndarray:
        """Return mask of pixels with valid hydraulic properties."""
        return (~np.isclose(self.field_capacity, self.hydraulic_nodata)) & (
            ~np.isclose(self.wilting_point, self.hydraulic_nodata)
        )