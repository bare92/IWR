from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


USDA_TEXTURE: dict[int, dict[str, float]] = {
    1: {"sand": 20.0, "clay": 60.0},
    2: {"sand": 10.0, "clay": 45.0},
    3: {"sand": 30.0, "clay": 50.0},
    4: {"sand": 10.0, "clay": 35.0},
    5: {"sand": 32.0, "clay": 34.0},
    6: {"sand": 8.0, "clay": 4.0},
    7: {"sand": 20.0, "clay": 15.0},
    8: {"sand": 55.0, "clay": 35.0},
    9: {"sand": 40.0, "clay": 20.0},
    10: {"sand": 60.0, "clay": 25.0},
    11: {"sand": 65.0, "clay": 10.0},
    12: {"sand": 80.0, "clay": 5.0},
    13: {"sand": 92.0, "clay": 3.0},
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


class IWRModel:
    """Simple beginner-friendly IWR model scaffold.

    This class focuses on:
    1) reading static input paths from JSON config files,
    2) exposing core raster attributes (including crop_fraction),
    3) converting soil texture to hydraulic properties,
    4) classifying daily phenology stage per pixel.
    """

    STAGE_INACTIVE = 0
    STAGE_GROWING = 1
    STAGE_MAXIMUM = 2
    STAGE_SENESCENCE = 3

    def __init__(
        self,
        model_inputs_config: str = "config/DAO_IWR_model_inputs.json",
        datasets_config: str = "config/DAO_IWR_datasets.json",
        tags_config: str = "config/DAO_IWR_tags.json",
    ) -> None:
        self.model_inputs_config = Path(model_inputs_config)
        self.datasets_config = Path(datasets_config)
        self.tags_config = Path(tags_config)

        self.static_inputs_paths: dict[str, str] = {}
        self.initial_condition: str = "watneeds_half_taw"
        self.start_date: date | None = None
        self.end_date: date | None = None

        # Raster attributes loaded by load_static_rasters().
        self.crop_fraction: np.ndarray | None = None
        self.soil_type: np.ndarray | None = None
        self.irrigated_mask: np.ndarray | None = None
        self.phenoe1: np.ndarray | None = None
        self.phenoe2: np.ndarray | None = None
        self.phenom1: np.ndarray | None = None
        self.phenom2: np.ndarray | None = None
        self.phenos1: np.ndarray | None = None
        self.phenos2: np.ndarray | None = None
        self.phenosen1: np.ndarray | None = None
        self.phenosen2: np.ndarray | None = None

        self._load_static_input_paths_from_config()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a JSON object: {path}")
        return data

    @staticmethod
    def _resolve_str(value: str, context: dict[str, Any], *, max_passes: int = 10) -> str:
        """Resolve placeholders like {KEY} and {KEY.subkey}."""
        for _ in range(max_passes):
            prev = value

            def replacer(match: re.Match[str]) -> str:
                path = match.group(1).split(".")
                obj: Any = context
                for part in path:
                    if isinstance(obj, dict) and part in obj:
                        obj = obj[part]
                    else:
                        return match.group(0)
                return str(obj) if not isinstance(obj, (dict, list)) else match.group(0)

            value = re.sub(r"\{([^}]+)\}", replacer, value)
            if value == prev:
                break

        return value

    def _resolve_obj(self, obj: Any, context: dict[str, Any]) -> Any:
        if isinstance(obj, str):
            return self._resolve_str(obj, context)
        if isinstance(obj, dict):
            return {k: self._resolve_obj(v, context) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_obj(item, context) for item in obj]
        return obj

    def _build_dataset_lookup(self) -> tuple[dict[str, str], dict[str, Any]]:
        datasets_data = self._read_json(self.datasets_config)
        tags_data = self._read_json(self.tags_config)

        tags_raw = tags_data.get("TAGS", {})
        if not isinstance(tags_raw, dict):
            raise ValueError("TAGS must be an object in tags config")

        resolved_tags = self._resolve_obj(tags_raw, tags_raw)

        datasets_root = datasets_data.get("DATASETS", {})
        if not isinstance(datasets_root, dict):
            raise ValueError("DATASETS must be an object in datasets config")

        lookup: dict[str, str] = {}
        for category_data in datasets_root.values():
            if not isinstance(category_data, dict):
                continue
            base_path = self._resolve_obj(category_data.get("path", ""), resolved_tags)
            if not isinstance(base_path, str):
                base_path = str(base_path)

            items = category_data.get("items", {})
            if not isinstance(items, dict):
                continue

            for name, item in items.items():
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename", "")
                if not filename:
                    continue
                folder = self._resolve_obj(item.get("folder", ""), resolved_tags)
                folder = folder if isinstance(folder, str) else str(folder)

                parts = [p for p in [base_path, folder, str(filename)] if p]
                lookup[name] = str(Path(*parts))

        return lookup, resolved_tags

    def _load_static_input_paths_from_config(self) -> None:
        model_inputs = self._read_json(self.model_inputs_config)
        model_cfg = model_inputs.get("WORKFLOW", {}).get("MODEL", {})
        if not isinstance(model_cfg, dict):
            raise ValueError("WORKFLOW.MODEL must be an object in model inputs config")

        static_inputs = model_cfg.get("static_inputs", {})
        if not isinstance(static_inputs, dict):
            raise ValueError("WORKFLOW.MODEL.static_inputs must be an object")

        dataset_lookup, resolved_tags = self._build_dataset_lookup()
        context = {"DATASETS": dataset_lookup, **resolved_tags}

        resolved_static = self._resolve_obj(static_inputs, context)
        if not isinstance(resolved_static, dict):
            raise ValueError("Resolved static_inputs must be an object")

        self.static_inputs_paths = {k: str(v) for k, v in resolved_static.items()}
        self.initial_condition = str(self.static_inputs_paths.get("initial_condition", "watneeds_half_taw"))

        time_range = model_cfg.get("time_range", {})
        if isinstance(time_range, dict):
            resolved_time_range = self._resolve_obj(time_range, context)
            if isinstance(resolved_time_range, dict):
                self.start_date = self._parse_iso_date_maybe(resolved_time_range.get("start"))
                self.end_date = self._parse_iso_date_maybe(resolved_time_range.get("end"))

    @staticmethod
    def _read_raster(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {path}")
        with rasterio.open(p) as src:
            return src.read()

    def load_static_rasters(self) -> None:
        """Load configured static rasters into object attributes."""
        self.crop_fraction = self._read_raster(self.static_inputs_paths["land_cover_fractional"])
        self.soil_type = self._read_raster(self.static_inputs_paths["soil_type"])
        self.irrigated_mask = self._read_raster(self.static_inputs_paths["irrigated_mask"])
        self.phenoe1 = self._read_raster(self.static_inputs_paths["phenoe1"])
        self.phenoe2 = self._read_raster(self.static_inputs_paths["phenoe2"])
        self.phenom1 = self._read_raster(self.static_inputs_paths["phenom1"])
        self.phenom2 = self._read_raster(self.static_inputs_paths["phenom2"])
        self.phenos1 = self._read_raster(self.static_inputs_paths["phenos1"])
        self.phenos2 = self._read_raster(self.static_inputs_paths["phenos2"])
        self.phenosen1 = self._read_raster(self.static_inputs_paths["phenosen1"])
        self.phenosen2 = self._read_raster(self.static_inputs_paths["phenosen2"])

    @staticmethod
    def _parse_iso_date_maybe(value: Any) -> date | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None

    @staticmethod
    def _to_date(value: date | str) -> date:
        if isinstance(value, date):
            return value
        return datetime.strptime(value, "%Y-%m-%d").date()

    @staticmethod
    def _single_band(arr: np.ndarray, name: str) -> np.ndarray:
        """Convert a raster to 2D by accepting either (rows, cols) or (1, rows, cols)."""
        data = np.asarray(arr)
        if data.ndim == 2:
            return data
        if data.ndim == 3 and data.shape[0] == 1:
            return data[0]
        raise ValueError(f"{name} must be 2D or single-band 3D; got shape {data.shape}")

    @staticmethod
    def _doy_to_dekad(doy: int) -> int:
        """Convert day-of-year to dekad index in [1, 36]."""
        return max(1, min(36, ((doy - 1) // 10) + 1))

    @staticmethod
    def _in_cyclic_interval(value: int, start: np.ndarray, end: np.ndarray, include_end: bool) -> np.ndarray:
        """Check if value belongs to [start, end] on a cyclic 1..36 scale (handles wrap-around)."""
        if include_end:
            non_wrap = (start <= end) & (value >= start) & (value <= end)
            wrap = (start > end) & ((value >= start) | (value <= end))
        else:
            non_wrap = (start <= end) & (value >= start) & (value < end)
            wrap = (start > end) & ((value >= start) | (value < end))
        return non_wrap | wrap

    def _classify_one_season(
        self,
        dekad: int,
        sos: np.ndarray,
        tom: np.ndarray,
        sen: np.ndarray,
        eos: np.ndarray,
    ) -> np.ndarray:
        """Classify stage for one season using SOS->TOM->SEN->EOS boundaries."""
        stage = np.full(sos.shape, self.STAGE_INACTIVE, dtype=np.uint8)

        valid = np.isfinite(sos) & np.isfinite(tom) & np.isfinite(sen) & np.isfinite(eos)
        valid &= (sos > 0) & (tom > 0) & (sen > 0) & (eos > 0)

        in_growing = valid & self._in_cyclic_interval(dekad, sos, tom, include_end=False)
        in_maximum = valid & self._in_cyclic_interval(dekad, tom, sen, include_end=False)
        in_senescence = valid & self._in_cyclic_interval(dekad, sen, eos, include_end=True)

        stage[in_growing] = self.STAGE_GROWING
        stage[in_maximum] = self.STAGE_MAXIMUM
        stage[in_senescence] = self.STAGE_SENESCENCE
        return stage

    @classmethod
    def stage_name_map(cls) -> dict[int, str]:
        return {
            cls.STAGE_INACTIVE: "inactive",
            cls.STAGE_GROWING: "growing",
            cls.STAGE_MAXIMUM: "maximum",
            cls.STAGE_SENESCENCE: "senescence",
        }

    def classify_phenology_stage_for_date(self, current_date: date | str) -> np.ndarray:
        """Return per-pixel stage map for one day.

        Input layers are interpreted as:
        - phenosen*: start of season (SOS)
        - phenom*: time of maximum (TOM)
        - phenos*: senescence start (SEN)
        - phenoe*: end of season (EOS)
        """
        if any(
            layer is None
            for layer in [
                self.phenoe1,
                self.phenoe2,
                self.phenom1,
                self.phenom2,
                self.phenos1,
                self.phenos2,
                self.phenosen1,
                self.phenosen2,
            ]
        ):
            raise ValueError("Phenology rasters are not loaded. Run load_static_rasters() first.")

        day = self._to_date(current_date)
        dekad = self._doy_to_dekad(day.timetuple().tm_yday)

        sos1 = self._single_band(self.phenosen1, "phenosen1")
        tom1 = self._single_band(self.phenom1, "phenom1")
        sen1 = self._single_band(self.phenos1, "phenos1")
        eos1 = self._single_band(self.phenoe1, "phenoe1")

        sos2 = self._single_band(self.phenosen2, "phenosen2")
        tom2 = self._single_band(self.phenom2, "phenom2")
        sen2 = self._single_band(self.phenos2, "phenos2")
        eos2 = self._single_band(self.phenoe2, "phenoe2")

        stage_1 = self._classify_one_season(dekad, sos1, tom1, sen1, eos1)
        stage_2 = self._classify_one_season(dekad, sos2, tom2, sen2, eos2)

        # Merge seasons by priority: maximum > senescence > growing > inactive.
        return np.maximum(stage_1, stage_2)

    def iter_daily_phenology_stages(
        self,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ):
        """Yield (date, stage_map) for every day in [start_date, end_date]."""
        start = self._to_date(start_date) if start_date is not None else self.start_date
        end = self._to_date(end_date) if end_date is not None else self.end_date

        if start is None or end is None:
            raise ValueError(
                "Start/end date missing. Pass start_date and end_date explicitly, "
                "or define resolvable WORKFLOW.MODEL.time_range in config."
            )
        if end < start:
            raise ValueError("end_date must be >= start_date")

        current = start
        while current <= end:
            yield current, self.classify_phenology_stage_for_date(current)
            current += timedelta(days=1)

    def build_daily_phenology_calendar(
        self,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> dict[date, np.ndarray]:
        """Build a dictionary of daily stage maps keyed by date."""
        return {
            day: stage_map
            for day, stage_map in self.iter_daily_phenology_stages(
                start_date=start_date,
                end_date=end_date,
            )
        }

    @staticmethod
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

    def hydraulic_from_soil_class(
        self,
        soil_class_id: int,
        org_mat_override: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Convert one USDA soil class id into (theta_wp, theta_fc, theta_s, ksat)."""
        if soil_class_id not in USDA_TEXTURE:
            raise ValueError(f"Unknown USDA soil class id: {soil_class_id}")

        tex = USDA_TEXTURE[soil_class_id]
        org_mat = USDA_DEFAULT_OM[soil_class_id] if org_mat_override is None else float(org_mat_override)

        return self.aquacrop_hydraulic_from_texture(
            sand_pct=float(tex["sand"]),
            clay_pct=float(tex["clay"]),
            org_mat=org_mat,
            df=1.0,
        )

    def build_hydraulic_layers_from_soil_type(
        self,
        org_mat_override: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Build theta_wp/theta_fc/theta_s/ksat rasters from the loaded soil_type raster."""
        if self.soil_type is None:
            raise ValueError("soil_type is not loaded. Run load_static_rasters() first.")

        soil = np.asarray(self.soil_type)
        if soil.ndim == 3 and soil.shape[0] == 1:
            soil = soil[0]
        if soil.ndim != 2:
            raise ValueError(f"soil_type must be 2D or single-band 3D; got shape {soil.shape}")

        th_wp = np.full(soil.shape, np.nan, dtype=np.float32)
        th_fc = np.full(soil.shape, np.nan, dtype=np.float32)
        th_s = np.full(soil.shape, np.nan, dtype=np.float32)
        ksat = np.full(soil.shape, np.nan, dtype=np.float32)

        valid_values = soil[np.isfinite(soil)]
        for raw_class in np.unique(valid_values):
            class_id = int(raw_class)
            if class_id == 0 or class_id not in USDA_TEXTURE:
                continue

            wp, fc, sat, ks = self.hydraulic_from_soil_class(class_id, org_mat_override=org_mat_override)
            mask = soil == raw_class
            th_wp[mask] = wp
            th_fc[mask] = fc
            th_s[mask] = sat
            ksat[mask] = ks

        return {
            "theta_wp": th_wp,
            "theta_fc": th_fc,
            "theta_s": th_s,
            "ksat_mm_day": ksat,
        }

    def summary(self) -> dict[str, Any]:
        """Small summary useful for quick checks while learning."""
        return {
            "model_inputs_config": str(self.model_inputs_config),
            "datasets_config": str(self.datasets_config),
            "tags_config": str(self.tags_config),
            "start_date": str(self.start_date) if self.start_date else None,
            "end_date": str(self.end_date) if self.end_date else None,
            "initial_condition": self.initial_condition,
            "loaded_paths": self.static_inputs_paths,
            "crop_fraction_loaded": self.crop_fraction is not None,
            "soil_type_loaded": self.soil_type is not None,
        }
