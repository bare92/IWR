
"""Daily soil water balance kernel for WATNEEDS-style IWR modeling.

This module provides the core single-day water-balance equations (FAO-56 with
WATNEEDS modifications). It is designed to be called by a time-series driver
(crop_calendar.TimeSeriesDriver) that orchestrates daily steps, manages crop
calendars, and aggregates outputs.

For full WATNEEDS implementation with crop calendars and seasonal aggregation,
see: crop_calendar.py and ISSUE_H_CROP_CALENDARS.md
"""

import numpy as np
import rasterio

from iwr_processing.iwr_layers import CropFraction, SoilLayer


def compute_taw(
    theta_fc: np.ndarray,
    theta_wp: np.ndarray,
    root_depth_m: float,
) -> np.ndarray:
    """Total available water in mm."""
    return 1000.0 * (theta_fc - theta_wp) * root_depth_m


def compute_raw(taw: np.ndarray, p: float) -> np.ndarray:
    """Readily available water in mm."""
    return p * taw


def compute_effective_precipitation(
    precipitation_mm: np.ndarray,
    reduction_pct: float = 5.0,
) -> np.ndarray:
    """Effective precipitation in mm after a fixed reduction factor."""
    factor = 1.0 - (reduction_pct / 100.0)
    return np.maximum(0.0, precipitation_mm * factor)


def compute_stress_coefficient(
    soil_moisture_mm: np.ndarray,
    mad_mm: np.ndarray,
    wp_mm: np.ndarray,
) -> np.ndarray:
    """Water-stress coefficient ks based on soil moisture, MAD and WP.

    ks = 1 for S >= MAD;
    ks = (S - WP) / (MAD - WP) for WP < S < MAD;
    ks = 0 for S <= WP.
    """
    ks = np.ones_like(soil_moisture_mm, dtype=np.float32)
    no_stress = soil_moisture_mm >= mad_mm
    stressed = (~no_stress) & (mad_mm > wp_mm)
    ks[stressed] = (
        (soil_moisture_mm[stressed] - wp_mm[stressed])
        / (mad_mm[stressed] - wp_mm[stressed])
    )
    return np.clip(ks, 0.0, 1.0).astype(np.float32)


def compute_watneeds_deep_percolation(
    st_prev_mm: np.ndarray,
    mad_mm: np.ndarray,
    taw_mm: np.ndarray,
    fmax_mm: np.ndarray,
) -> np.ndarray:
    """WATNEEDS deep percolation term from previous soil moisture.

    D_t = Fmax * (S_(t-1) - MAD) / (TAW - MAD), clipped to [0, Fmax],
    and set to zero below MAD.
    """
    ratio = np.zeros_like(st_prev_mm, dtype=np.float32)
    valid_denom = (taw_mm - mad_mm) > 0
    ratio[valid_denom] = (
        (st_prev_mm[valid_denom] - mad_mm[valid_denom])
        / (taw_mm[valid_denom] - mad_mm[valid_denom])
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.maximum(0.0, fmax_mm * ratio).astype(np.float32)


# Default maximum infiltration/deep-percolation capacity by USDA texture class.
# Units: mm/day.
DEFAULT_FMAX_MM_DAY_BY_USDA_CLASS: dict[int, float] = {
    1: 2.0,   # Clay heavy
    2: 3.0,   # Silty clay
    3: 2.5,   # Clay
    4: 5.0,   # Silty clay loam
    5: 6.0,   # Clay loam
    6: 10.0,  # Silt
    7: 12.0,  # Silt loam
    8: 7.0,   # Sandy clay
    9: 15.0,  # Loam
    10: 10.0, # Sandy clay loam
    11: 20.0, # Sandy loam
    12: 30.0, # Loamy sand
    13: 40.0, # Sand
}


def watneeds_green_step(
    s_prev_mm: np.ndarray,
    precipitation_mm: np.ndarray,
    etc_mm: np.ndarray,
    taw_mm: np.ndarray,
    p: np.ndarray | float,
    fmax_mm_day: np.ndarray,
    peff_coeff: float = 0.95,
    dt_days: float = 1.0,
    irrigated_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """WATNEEDS-style daily green/blue crop water requirement step.

    Pure vectorized function implementing the WATNEEDS daily water balance with
    stress-dependent green water (ET under soil moisture stress), blue water demand
    (remaining ETc on irrigated land), and deep percolation.

    State:
        s_prev_mm and s_next_mm are available root-zone water storage above wilting
        point, bounded in [0, TAW].

    Green balance:
        Peff = peff_coeff * P * dt_days
        surface_runoff = (1 - peff_coeff) * P * dt_days
        threshold = (1 - p) * TAW = RAW threshold for stress
        Ks = 1 if S_prev >= threshold; else S_prev / threshold (linear stress)
        GW = Ks * ETc
        DP = Fmax * (S_prev - threshold) / (TAW - threshold), clipped to [0, Fmax*dt_days]
        S_next = S_prev + Peff - GW - DP
        If S_next < 0: scale GW and DP proportionally down
        If S_next > TAW: overflow_runoff = S_next - TAW, S_next = TAW

    Demand:
        BW = max(ETc - GW, 0) on irrigated pixels only.
        BW = 0 on rainfed pixels.

    Args:
        s_prev_mm: Soil moisture at start of day (mm), shape (rows, cols).
        precipitation_mm: Daily precipitation (mm), shape (rows, cols).
        etc_mm: Crop evapotranspiration rate (mm/day), shape (rows, cols) or scalar.
        taw_mm: Total available water (mm), shape (rows, cols).
        p: Depletion fraction for RAW, scalar or array. Typical range [0.3, 0.6].
        fmax_mm_day: Maximum deep percolation rate (mm/day), shape (rows, cols).
        peff_coeff: Effective precipitation coefficient, default 0.95 (5% runoff).
        dt_days: Timestep in days, default 1.0.
        irrigated_mask: Boolean array where True = irrigated. If None, no blue water.
        valid_mask: Boolean array where True = valid pixels. If None, all valid.

    Returns:
        Dict with keys:
            - s_next_mm: Soil moisture at end of day (mm), clipped [0, TAW]
            - green_et_mm: Green ET (actual ET under stress) (mm)
            - deep_perc_mm: Deep percolation loss (mm)
            - blue_water_mm: Blue water requirement = max(ETc - GW, 0) (mm)
            - ks: Stress coefficient [0, 1]
            - surface_runoff_mm: Runoff from precipitation (1-peff_coeff) * P
            - overflow_runoff_mm: Runoff from saturation (S_next > TAW)
            - total_runoff_mm: surface_runoff + overflow_runoff
            - peff_mm: Effective precipitation
            - available_mm: S_prev + Peff (available for ET and DP)
    """
    # Ensure dtypes
    s_prev_mm = np.asarray(s_prev_mm, dtype=np.float32)
    precipitation_mm = np.asarray(precipitation_mm, dtype=np.float32)
    etc_mm = np.asarray(etc_mm, dtype=np.float32)
    taw_mm = np.asarray(taw_mm, dtype=np.float32)
    fmax_mm_day = np.asarray(fmax_mm_day, dtype=np.float32)
    p = float(p) if np.isscalar(p) else np.asarray(p, dtype=np.float32)

    # Compute effective precipitation and surface runoff
    peff_mm = peff_coeff * precipitation_mm * float(dt_days)
    surface_runoff_mm = (1.0 - peff_coeff) * precipitation_mm * float(dt_days)

    # Total ETc demand over timestep
    etc_total = np.maximum(etc_mm, 0.0) * float(dt_days)

    # Stress threshold (RAW) = (1 - p) * TAW
    threshold = np.maximum((1.0 - p) * taw_mm, 0.0)

    # Stress coefficient Ks: linear from 0 at WP to 1 at RAW threshold
    ks = np.ones_like(s_prev_mm, dtype=np.float32)
    stress_mask = s_prev_mm < threshold
    ks[stress_mask] = np.divide(
        s_prev_mm[stress_mask],
        threshold[stress_mask],
        out=np.zeros_like(s_prev_mm[stress_mask]),
        where=threshold[stress_mask] > 0.0,
    )
    ks = np.clip(ks, 0.0, 1.0).astype(np.float32)

    # Green ET (actual ET under stress)
    green_et_mm = ks * etc_total

    # Deep percolation: linear from S_prev = threshold to Fmax
    denom = taw_mm - threshold
    deep_perc_mm = np.where(
        (s_prev_mm >= threshold) & (denom > 0.0),
        fmax_mm_day * float(dt_days) * (s_prev_mm - threshold) / denom,
        0.0,
    ).astype(np.float32)
    deep_perc_mm = np.clip(deep_perc_mm, 0.0, fmax_mm_day * float(dt_days))

    # Water available for ET and DP
    available_mm = s_prev_mm + peff_mm
    losses_mm = green_et_mm + deep_perc_mm

    # If losses exceed available, scale both down proportionally
    negative_balance = available_mm < losses_mm
    scale_factor = np.ones_like(available_mm, dtype=np.float32)
    scale_factor[negative_balance & (losses_mm > 0.0)] = (
        available_mm[negative_balance & (losses_mm > 0.0)]
        / losses_mm[negative_balance & (losses_mm > 0.0)]
    )
    scale_factor = np.clip(scale_factor, 0.0, 1.0)

    green_et_mm = (green_et_mm * scale_factor).astype(np.float32)
    deep_perc_mm = (deep_perc_mm * scale_factor).astype(np.float32)

    # Update soil moisture and handle overflow
    s_after_mm = available_mm - green_et_mm - deep_perc_mm
    overflow_runoff_mm = np.maximum(s_after_mm - taw_mm, 0.0).astype(np.float32)
    s_next_mm = np.clip(s_after_mm, 0.0, taw_mm).astype(np.float32)

    # Blue water: remaining crop water demand on irrigated pixels
    blue_water_mm = np.maximum(etc_total - green_et_mm, 0.0).astype(np.float32)
    if irrigated_mask is not None:
        blue_water_mm = np.where(irrigated_mask, blue_water_mm, 0.0).astype(np.float32)

    # Total runoff
    total_runoff_mm = (surface_runoff_mm + overflow_runoff_mm).astype(np.float32)

    # Apply valid mask if provided
    if valid_mask is not None:
        s_next_mm = np.where(valid_mask, s_next_mm, 0.0).astype(np.float32)
        green_et_mm = np.where(valid_mask, green_et_mm, 0.0).astype(np.float32)
        deep_perc_mm = np.where(valid_mask, deep_perc_mm, 0.0).astype(np.float32)
        blue_water_mm = np.where(valid_mask, blue_water_mm, 0.0).astype(np.float32)
        ks = np.where(valid_mask, ks, 0.0).astype(np.float32)
        total_runoff_mm = np.where(valid_mask, total_runoff_mm, 0.0).astype(np.float32)

    return {
        "s_next_mm": s_next_mm,
        "green_et_mm": green_et_mm,
        "deep_perc_mm": deep_perc_mm,
        "blue_water_mm": blue_water_mm,
        "ks": ks,
        "surface_runoff_mm": surface_runoff_mm,
        "overflow_runoff_mm": overflow_runoff_mm,
        "total_runoff_mm": total_runoff_mm,
        "peff_mm": peff_mm,
        "available_mm": available_mm,
    }


class IWRModel:
    def __init__(
        self,
        time_range: str | None,
        land_cover_path: str,
        soil_path: str,
        crop_parameter_csv: str | None = None,
        initial_condition: str = "watneeds_half_taw",
        initial_saturation: np.ndarray | str | None = None,
        peff_reduction_pct: float = 5.0,
        taw_layer: np.ndarray | str | None = None,
        smax_layer: np.ndarray | str | None = None,
        fmax_layer: np.ndarray | str | None = None,
    ):
        """Initialize IWR model.
        
        Args:
            time_range: Optional date range descriptor (currently unused).
            land_cover_path: Path to multiband crop-fraction GeoTIFF.
            soil_path: Path to USDA soil-class raster.
            crop_parameter_csv: Optional path to FAO-56 crop parameter table.
            initial_condition: One of 'watneeds_half_taw', 'field_capacity',
                or 'saturation_raster'.
            initial_saturation: Optional raster path or array for 'saturation_raster'
                mode.
            peff_reduction_pct: Effective precipitation reduction (0-100, default 5).
            taw_layer: Optional raster path or array (mm) for WATNEEDS-compliant direct
                TAW input. If provided, overrides pedotransfer computation from FC/WP.
            smax_layer: Optional raster path or array (mm) for maximum soil storage
                capacity. Currently stored but not yet used in balance equations.
            fmax_layer: Optional raster path or array (mm/day) for WATNEEDS-compliant
                maximum percolation / infiltration capacity. Preferred over USDA fallback.
        """
        self.time_range = time_range
        self.crop_fractions = CropFraction.from_geotiff(
            land_cover_path,
            crop_parameter_csv=crop_parameter_csv,
        )
        self.soil = SoilLayer.from_geotiff(
            soil_path,
            taw_mm=taw_layer,
            smax_mm=smax_layer,
        )

        self.initial_condition = initial_condition
        self.initial_saturation = self._load_initial_saturation(initial_saturation)
        self.peff_reduction_pct = float(peff_reduction_pct)
        self.fmax_layer = fmax_layer
        self._validate_initialization()

    def _load_initial_saturation(
        self,
        initial_saturation: np.ndarray | str | None,
    ) -> np.ndarray | None:
        """Load optional initial saturation raster as relative values in [0, 1]."""
        if initial_saturation is None:
            return None

        if isinstance(initial_saturation, str):
            with rasterio.open(initial_saturation) as src:
                saturation = src.read(1).astype(np.float32)
        else:
            saturation = np.asarray(initial_saturation, dtype=np.float32)

        if saturation.shape != self.soil.field_capacity.shape:
            raise ValueError(
                "initial_saturation shape must match soil shape "
                f"{self.soil.field_capacity.shape}, got {saturation.shape}."
            )

        return saturation

    def _validate_initialization(self) -> None:
        """Validate initialization options for water-balance state."""
        valid_conditions = {"watneeds_half_taw", "field_capacity", "saturation_raster"}
        if self.initial_condition not in valid_conditions:
            raise ValueError(
                "initial_condition must be one of "
                f"{sorted(valid_conditions)}, got '{self.initial_condition}'."
            )

        if self.initial_condition == "saturation_raster" and self.initial_saturation is None:
            raise ValueError(
                "initial_condition='saturation_raster' requires initial_saturation "
                "as a raster path or numpy array."
            )

        if not (0.0 <= self.peff_reduction_pct <= 100.0):
            raise ValueError(
                "peff_reduction_pct must be between 0 and 100, "
                f"got {self.peff_reduction_pct}."
            )

    def _load_precipitation(
        self,
        precipitation: np.ndarray | str,
    ) -> np.ndarray:
        """Load precipitation raster/array in mm for the model grid."""
        if isinstance(precipitation, str):
            with rasterio.open(precipitation) as src:
                precip = src.read(1).astype(np.float32)
        else:
            precip = np.asarray(precipitation, dtype=np.float32)

        if precip.shape != self.soil.field_capacity.shape:
            raise ValueError(
                "precipitation shape must match soil shape "
                f"{self.soil.field_capacity.shape}, got {precip.shape}."
            )

        return precip

    def _as_grid_array(
        self,
        value: np.ndarray | str | float | int,
        name: str,
    ) -> np.ndarray:
        """Convert scalar, array, or raster path to a model-grid array."""
        if isinstance(value, str):
            with rasterio.open(value) as src:
                arr = src.read(1).astype(np.float32)
        elif np.isscalar(value):
            arr = np.full(self.soil.field_capacity.shape, float(value), dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)

        if arr.shape != self.soil.field_capacity.shape:
            raise ValueError(
                f"{name} shape must match soil shape "
                f"{self.soil.field_capacity.shape}, got {arr.shape}."
            )

        return arr

    def _default_fmax_mm_day_from_soil(self) -> np.ndarray:
        """Build Fmax raster from USDA soil texture classes (mm/day)."""
        fmax = np.zeros(self.soil.soil_type.shape, dtype=np.float32)
        valid_mask = self.soil.hydraulic_mask()

        for class_id, value in DEFAULT_FMAX_MM_DAY_BY_USDA_CLASS.items():
            class_mask = np.isclose(self.soil.soil_type, class_id)
            fmax[class_mask] = float(value)

        # Keep zeros outside valid hydraulic pixels.
        fmax[~valid_mask] = 0.0
        return fmax

    def get_effective_precipitation(
        self,
        precipitation: np.ndarray | str,
        reduction_pct: float | None = None,
    ) -> np.ndarray:
        """Return effective precipitation (Peff) in mm for each pixel."""
        precip = self._load_precipitation(precipitation)
        reduction = self.peff_reduction_pct if reduction_pct is None else float(reduction_pct)
        if not (0.0 <= reduction <= 100.0):
            raise ValueError(f"reduction_pct must be between 0 and 100, got {reduction}.")

        valid_mask = self.soil.hydraulic_mask()
        peff = compute_effective_precipitation(precip, reduction_pct=reduction).astype(np.float32)
        peff[~valid_mask] = self.soil.hydraulic_nodata
        return peff

    def get_crop_initial_saturation(self) -> np.ndarray:
        """Return effective initial relative saturation in [0, 1] for each pixel."""
        valid_mask = self.soil.hydraulic_mask()

        if self.initial_condition == "watneeds_half_taw":
            sat = np.full_like(self.soil.field_capacity, 0.5, dtype=np.float32)
        elif self.initial_condition == "field_capacity":
            sat = np.ones_like(self.soil.field_capacity, dtype=np.float32)
        else:
            sat = self.initial_saturation.astype(np.float32)
            max_sat = float(np.nanmax(sat)) if sat.size else 0.0
            # Allow users to pass saturation as either 0-1 or 0-100 values.
            if max_sat > 1.0:
                sat = sat / 100.0
            sat = np.clip(sat, 0.0, 1.0)

        sat[~valid_mask] = 0.0
        return sat

    def get_crop_initial_depletion_mm(
        self,
        crop: str,
        root_depth_use: str = "max",
    ) -> np.ndarray:
        """Return initial root-zone depletion Dr0 in mm for one crop.

        Dr0 = (1 - relative_saturation) * TAW.
        """
        taw = self.get_crop_taw(crop, root_depth_use=root_depth_use)
        sat = self.get_crop_initial_saturation()
        dr0 = ((1.0 - sat) * taw).astype(np.float32)
        dr0[~self.soil.hydraulic_mask()] = self.soil.hydraulic_nodata
        return dr0

    def get_crop_initial_soil_water_mm(
        self,
        crop: str,
        root_depth_use: str = "max",
    ) -> np.ndarray:
        """Return initial available soil water in mm for one crop.

        This quantity is relative to the wilting point and bounded by TAW.
        """
        taw = self.get_crop_taw(crop, root_depth_use=root_depth_use)
        sat = self.get_crop_initial_saturation()
        sw0 = (sat * taw).astype(np.float32)
        sw0[~self.soil.hydraulic_mask()] = self.soil.hydraulic_nodata
        return sw0

    def get_crop_taw(self, crop: str, root_depth_use: str = "max") -> np.ndarray:
        """Return TAW raster for one crop.
        
        If external taw_mm raster was provided (WATNEEDS direct input),
        returns that directly. Otherwise computes from soil FC/WP and
        crop root depth (pedotransfer fallback).
        """
        # WATNEEDS-compliant: use external gridded soil storage if available
        if self.soil.taw_mm is not None:
            taw = self.soil.taw_mm.copy().astype(np.float32)
            taw[~self.soil.hydraulic_mask()] = self.soil.hydraulic_nodata
            return taw
        
        # Fallback: compute from pedotransfer (local adaptation mode)
        root_depth_m = self.crop_fractions.get_crop_root_depth(crop, use=root_depth_use)
        taw = compute_taw(
            self.soil.field_capacity,
            self.soil.wilting_point,
            root_depth_m,
        ).astype(np.float32)
        taw[~self.soil.hydraulic_mask()] = self.soil.hydraulic_nodata
        return taw

    def get_crop_raw(
        self,
        crop: str,
        etc_mm_day: float | None = None,
        root_depth_use: str = "max",
    ) -> np.ndarray:
        """Return RAW raster for one crop using FAO-56 p and TAW."""
        taw = self.get_crop_taw(crop, root_depth_use=root_depth_use)
        p_value = self.crop_fractions.get_crop_p(crop, etc_mm_day=etc_mm_day)
        raw = compute_raw(taw, p_value).astype(np.float32)
        raw[~self.soil.hydraulic_mask()] = self.soil.hydraulic_nodata
        return raw

    def green_water_step(
        self,
        crop: str,
        s_prev_mm: np.ndarray | str,
        precipitation_mm: np.ndarray | str,
        etc_mm: np.ndarray | str | float | int,
        fmax_mm_day: np.ndarray | str | float | int | None = None,
        irrigated_mask: np.ndarray | str | None = None,
        peff_coeff: float = 0.95,
        dt_days: float = 1.0,
        root_depth_use: str = "max",
    ) -> dict[str, np.ndarray]:
        """Wrapper for pure watneeds_green_step function.

        Simplified interface: takes crop name, loads soil parameters, calls
        the pure vectorized watneeds_green_step() function.

        Args:
            crop: Crop identifier (loads TAW and p from crop parameters).
            s_prev_mm: Soil moisture at start of day (mm).
            precipitation_mm: Daily precipitation (mm).
            etc_mm: Crop evapotranspiration (mm/day).
            fmax_mm_day: Maximum deep percolation (mm/day). If None, uses USDA.
            irrigated_mask: Boolean mask of irrigated pixels.
            peff_coeff: Effective precipitation coefficient (default 0.95).
            dt_days: Timestep in days (default 1.0).
            root_depth_use: Crop root depth selection ('min', 'max', 'mean').

        Returns:
            Dict from watneeds_green_step with keys:
                s_next_mm, green_et_mm, deep_perc_mm, blue_water_mm, ks,
                surface_runoff_mm, overflow_runoff_mm, total_runoff_mm, peff_mm, available_mm.
        """
        # Load soil/crop parameters
        s_prev = self._as_grid_array(s_prev_mm, name="s_prev_mm")
        precip = self._load_precipitation(precipitation_mm)
        etc = self._as_grid_array(etc_mm, name="etc_mm")
        taw = self.get_crop_taw(crop, root_depth_use=root_depth_use)
        p_value = self.crop_fractions.get_crop_p(crop, etc_mm_day=float(np.nanmean(etc)))
        
        # Get Fmax
        if fmax_mm_day is None:
            if self.fmax_layer is not None:
                fmax = self._as_grid_array(self.fmax_layer, name="fmax_layer")
            else:
                fmax = self._default_fmax_mm_day_from_soil()
        else:
            fmax = self._as_grid_array(fmax_mm_day, name="fmax_mm_day")

        # Get irrigated mask
        if irrigated_mask is not None:
            if isinstance(irrigated_mask, str):
                with rasterio.open(irrigated_mask) as src:
                    irr_mask = src.read(1) > 0
            else:
                irr_mask = np.asarray(irrigated_mask, dtype=bool)
        else:
            irr_mask = None

        # Valid mask (hydraulic layer)
        valid_mask = self.soil.hydraulic_mask()

        # Call pure function
        return watneeds_green_step(
            s_prev_mm=s_prev,
            precipitation_mm=precip,
            etc_mm=etc,
            taw_mm=taw,
            p=p_value,
            fmax_mm_day=fmax,
            peff_coeff=peff_coeff,
            dt_days=dt_days,
            irrigated_mask=irr_mask,
            valid_mask=valid_mask,
        )

    def soil_water_balance_step(
        self,
        crop: str,
        st_prev_mm: np.ndarray | str,
        precipitation_mm_day: np.ndarray | str,
        etc_mm_day: np.ndarray | str | float | int | None = None,
        et0_mm_day: np.ndarray | str | float | int | None = None,
        kc: np.ndarray | str | float | int | None = None,
        peff_partition_coeff: np.ndarray | str | float | int = 1.0,
        irrigation_mask: np.ndarray | str | None = None,
        is_irrigated: bool = True,
        d_t_mm_day: np.ndarray | str | float | int | None = None,
        r_t_mm_day: np.ndarray | str | float | int | None = None,
        fmax_mm_day: np.ndarray | str | float | int | None = None,
        root_depth_use: str = "max",
        dt_days: float = 1.0,
        peff_reduction_pct: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Default daily balance: pure WATNEEDS green-water formulation."""
        return self.watneeds_soil_water_balance_step(
            crop=crop,
            st_prev_mm=st_prev_mm,
            precipitation_mm_day=precipitation_mm_day,
            etc_mm_day=etc_mm_day,
            et0_mm_day=et0_mm_day,
            kc=kc,
            peff_partition_coeff=peff_partition_coeff,
            is_irrigated=is_irrigated,
            irrigation_mask=irrigation_mask,
            fmax_mm_day=fmax_mm_day,
            fmax_layer=self.fmax_layer,
            root_depth_use=root_depth_use,
            dt_days=dt_days,
            peff_reduction_pct=peff_reduction_pct,
        )

    def watneeds_soil_water_balance_step(
        self,
        crop: str,
        st_prev_mm: np.ndarray | str,
        precipitation_mm_day: np.ndarray | str,
        etc_mm_day: np.ndarray | str | float | int | None = None,
        et0_mm_day: np.ndarray | str | float | int | None = None,
        kc: np.ndarray | str | float | int | None = None,
        peff_partition_coeff: np.ndarray | str | float | int = 1.0,
        is_irrigated: bool = True,
        irrigation_mask: np.ndarray | str | None = None,
        fmax_mm_day: np.ndarray | str | float | int | None = None,
        fmax_layer: np.ndarray | str | None = None,
        root_depth_use: str = "max",
        dt_days: float = 1.0,
        peff_reduction_pct: float | None = None,
    ) -> dict[str, np.ndarray]:
        """WATNEEDS-like one-day soil balance and green/blue ET partition.

        This method keeps the soil balance as a green-water balance only:

        S_t = S_(t-1) + Peff - ETa_green - D - R

        Blue-water requirement is then computed as unmet crop ET demand:

        BW = max(ETc - ETa_green, 0)

        and is masked to zero where crops are non-irrigated.
        """
        if dt_days <= 0:
            raise ValueError(f"dt_days must be > 0, got {dt_days}.")

        valid_mask = self.soil.hydraulic_mask()

        st_prev = self._as_grid_array(st_prev_mm, name="st_prev_mm")
        st_prev = np.maximum(st_prev, 0.0).astype(np.float32)

        precip = self._load_precipitation(precipitation_mm_day)
        peff_base = self.get_effective_precipitation(precip, reduction_pct=peff_reduction_pct)
        peff_coeff = self._as_grid_array(peff_partition_coeff, name="peff_partition_coeff")
        peff = np.maximum(0.0, peff_base * peff_coeff).astype(np.float32)
        surface_runoff_term = (
            np.maximum(np.maximum(precip, 0.0) - np.maximum(peff_base, 0.0), 0.0)
            * float(dt_days)
        ).astype(np.float32)

        if etc_mm_day is not None:
            etc_rate = self._as_grid_array(etc_mm_day, name="etc_mm_day")
        else:
            if et0_mm_day is None or kc is None:
                raise ValueError(
                    "Provide either etc_mm_day, or both et0_mm_day and kc "
                    "to compute ETc = ET0 * Kc."
                )
            et0 = self._as_grid_array(et0_mm_day, name="et0_mm_day")
            kc_arr = self._as_grid_array(kc, name="kc")
            etc_rate = (et0 * kc_arr).astype(np.float32)

        etc_term = (np.maximum(etc_rate, 0.0) * float(dt_days)).astype(np.float32)

        taw = self.get_crop_taw(crop, root_depth_use=root_depth_use)
        raw = self.get_crop_raw(crop, etc_mm_day=float(np.nanmean(etc_rate)), root_depth_use=root_depth_use)
        mad_mm = np.maximum(taw - raw, 0.0).astype(np.float32)
        wp_mm = np.zeros_like(taw, dtype=np.float32)

        ks = compute_stress_coefficient(st_prev, mad_mm=mad_mm, wp_mm=wp_mm)
        eta_green = (ks * etc_term).astype(np.float32)

        if fmax_mm_day is None:
            fmax_source = self.fmax_layer if fmax_layer is None else fmax_layer
            if fmax_source is None:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "fmax_mm_day and fmax_layer both None; using USDA class lookup as fallback. "
                    "For WATNEEDS mode, provide Fmax from external gridded soil data."
                )
                fmax_term = self._default_fmax_mm_day_from_soil() * float(dt_days)
            else:
                fmax_layer_arr = self._as_grid_array(fmax_source, name="fmax_layer")
                fmax_term = (np.maximum(fmax_layer_arr, 0.0) * float(dt_days)).astype(np.float32)
        else:
            fmax_term = (
                np.maximum(self._as_grid_array(fmax_mm_day, name="fmax_mm_day"), 0.0)
                * float(dt_days)
            ).astype(np.float32)

        available = st_prev + peff

        # D from WATNEEDS linear function of previous storage and Fmax.
        provisional_d = compute_watneeds_deep_percolation(
            st_prev_mm=st_prev,
            mad_mm=mad_mm,
            taw_mm=np.maximum(taw, 0.0),
            fmax_mm=fmax_term,
        )

        # If available - ETa - D < 0, scale ETa and D proportionally to close balance.
        total_losses = eta_green + provisional_d
        scale = np.ones_like(available, dtype=np.float32)
        neg_mask = (available - total_losses) < 0.0
        valid_losses = neg_mask & (total_losses > 0.0)
        scale[valid_losses] = available[valid_losses] / total_losses[valid_losses]
        scale = np.clip(scale, 0.0, 1.0)

        eta_green = eta_green * scale
        d_term = provisional_d * scale

        balance_after_d = available - eta_green - d_term
        r_term = np.maximum(balance_after_d - np.maximum(taw, 0.0), 0.0).astype(np.float32)
        st_next = np.clip(balance_after_d - r_term, 0.0, np.maximum(taw, 0.0)).astype(np.float32)
        total_runoff = (surface_runoff_term + r_term).astype(np.float32)

        # Blue water requirement is unmet ET demand and does not alter green-state storage.
        blue_water_req = np.maximum(etc_term - eta_green, 0.0).astype(np.float32)
        if is_irrigated:
            if irrigation_mask is None:
                irrigated_pixels = valid_mask
            else:
                mask_arr = self._as_grid_array(irrigation_mask, name="irrigation_mask")
                irrigated_pixels = (mask_arr > 0) & valid_mask
            blue_water_req[~irrigated_pixels] = 0.0
        else:
            blue_water_req[:] = 0.0

        nodata = self.soil.hydraulic_nodata
        for arr in (
            peff,
            etc_term,
            ks,
            eta_green,
            d_term,
            r_term,
            surface_runoff_term,
            total_runoff,
            st_next,
            blue_water_req,
            fmax_term,
            mad_mm,
            taw,
            raw,
        ):
            arr[~valid_mask] = nodata

        return {
            "soil_moisture_next_mm": st_next,
            "effective_precipitation_mm": peff,
            "etc_mm": etc_term,
            "eta_green_mm": eta_green,
            "blue_water_requirement_mm": blue_water_req,
            "ks": ks,
            "d_t_mm": d_term,
            "r_t_mm": r_term,
            "overflow_runoff_mm": r_term,
            "surface_runoff_mm": surface_runoff_term,
            "total_runoff_mm": total_runoff,
            "fmax_mm": fmax_term,
            "raw_depletion_mm": raw,
            "stress_threshold_mm": mad_mm,
            "taw_mm": taw,
        }

    def watneeds_spinup(
        self,
        crop: str,
        precipitation_mm_day_series: list[np.ndarray | str],
        etc_mm_day_series: list[np.ndarray | str | float | int] | None = None,
        et0_mm_day_series: list[np.ndarray | str | float | int] | None = None,
        kc_series: list[np.ndarray | str | float | int] | None = None,
        peff_partition_coeff_series: list[np.ndarray | str | float | int] | None = None,
        is_irrigated: bool = True,
        irrigation_mask: np.ndarray | str | None = None,
        fmax_mm_day: np.ndarray | str | float | int | None = None,
        root_depth_use: str = "max",
        dt_days: float = 1.0,
        peff_reduction_pct: float | None = None,
        spinup_years: int = 3,
        initial_soil_moisture_mm: np.ndarray | str | float | int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run WATNEEDS spin-up over repeated daily forcing series.

        Defaults to 3-year spin-up and starts from 50% TAW unless
        initial_soil_moisture_mm is provided.
        """
        if spinup_years < 1:
            raise ValueError(f"spinup_years must be >= 1, got {spinup_years}.")
        if not precipitation_mm_day_series:
            raise ValueError("precipitation_mm_day_series must contain at least one day.")

        n_days = len(precipitation_mm_day_series)

        def _series_value(series: list, idx: int, name: str):
            if len(series) != n_days:
                raise ValueError(f"{name} length must be {n_days}, got {len(series)}.")
            return series[idx]

        if initial_soil_moisture_mm is None:
            st = self.get_crop_initial_soil_water_mm(crop, root_depth_use=root_depth_use)
        else:
            st = np.maximum(
                self._as_grid_array(initial_soil_moisture_mm, name="initial_soil_moisture_mm"),
                0.0,
            ).astype(np.float32)

        last_step: dict[str, np.ndarray] | None = None
        for _ in range(spinup_years):
            for day_idx in range(n_days):
                step_kwargs = {
                    "crop": crop,
                    "st_prev_mm": st,
                    "precipitation_mm_day": precipitation_mm_day_series[day_idx],
                    "peff_partition_coeff": (
                        1.0
                        if peff_partition_coeff_series is None
                        else _series_value(peff_partition_coeff_series, day_idx, "peff_partition_coeff_series")
                    ),
                    "is_irrigated": is_irrigated,
                    "irrigation_mask": irrigation_mask,
                    "fmax_mm_day": fmax_mm_day,
                    "root_depth_use": root_depth_use,
                    "dt_days": dt_days,
                    "peff_reduction_pct": peff_reduction_pct,
                }

                if etc_mm_day_series is not None:
                    step_kwargs["etc_mm_day"] = _series_value(etc_mm_day_series, day_idx, "etc_mm_day_series")
                else:
                    if et0_mm_day_series is None or kc_series is None:
                        raise ValueError(
                            "Provide etc_mm_day_series, or both et0_mm_day_series "
                            "and kc_series for spin-up."
                        )
                    step_kwargs["et0_mm_day"] = _series_value(et0_mm_day_series, day_idx, "et0_mm_day_series")
                    step_kwargs["kc"] = _series_value(kc_series, day_idx, "kc_series")

                last_step = self.watneeds_soil_water_balance_step(**step_kwargs)
                st = last_step["soil_moisture_next_mm"]

        if last_step is None:
            raise RuntimeError("Spin-up failed to run any timestep.")

        return {
            "soil_moisture_spinup_end_mm": st,
            "soil_moisture_last_step_mm": last_step["soil_moisture_next_mm"],
            "eta_green_last_step_mm": last_step["eta_green_mm"],
            "blue_water_last_step_mm": last_step["blue_water_requirement_mm"],
        }

    def operational_irrigation_balance_step(
        self,
        crop: str,
        st_prev_mm: np.ndarray | str,
        precipitation_mm_day: np.ndarray | str,
        irrigation_applied_mm_day: np.ndarray | str | float | int,
        etc_mm_day: np.ndarray | str | float | int | None = None,
        et0_mm_day: np.ndarray | str | float | int | None = None,
        kc: np.ndarray | str | float | int | None = None,
        peff_partition_coeff: np.ndarray | str | float | int = 1.0,
        fmax_mm_day: np.ndarray | str | float | int | None = None,
        fmax_layer: np.ndarray | str | None = None,
        root_depth_use: str = "max",
        dt_days: float = 1.0,
        peff_reduction_pct: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Optional operational mode with irrigation added to storage state.

        This is intentionally separate from WATNEEDS demand accounting.
        """
        if dt_days <= 0:
            raise ValueError(f"dt_days must be > 0, got {dt_days}.")

        valid_mask = self.soil.hydraulic_mask()
        st_prev = np.maximum(self._as_grid_array(st_prev_mm, name="st_prev_mm"), 0.0).astype(np.float32)

        precip = self._load_precipitation(precipitation_mm_day)
        peff_base = self.get_effective_precipitation(precip, reduction_pct=peff_reduction_pct)
        peff_coeff = self._as_grid_array(peff_partition_coeff, name="peff_partition_coeff")
        peff_term = (np.maximum(0.0, peff_base * peff_coeff) * float(dt_days)).astype(np.float32)
        surface_runoff_term = (
            np.maximum(np.maximum(precip, 0.0) - np.maximum(peff_base, 0.0), 0.0)
            * float(dt_days)
        ).astype(np.float32)

        irrigation_term = (
            np.maximum(self._as_grid_array(irrigation_applied_mm_day, name="irrigation_applied_mm_day"), 0.0)
            * float(dt_days)
        ).astype(np.float32)

        if etc_mm_day is not None:
            etc_rate = self._as_grid_array(etc_mm_day, name="etc_mm_day")
        else:
            if et0_mm_day is None or kc is None:
                raise ValueError(
                    "Provide either etc_mm_day, or both et0_mm_day and kc "
                    "to compute ETc = ET0 * Kc."
                )
            etc_rate = self._as_grid_array(et0_mm_day, name="et0_mm_day") * self._as_grid_array(kc, name="kc")

        etc_term = (np.maximum(etc_rate, 0.0) * float(dt_days)).astype(np.float32)
        taw = self.get_crop_taw(crop, root_depth_use=root_depth_use)
        raw = self.get_crop_raw(crop, etc_mm_day=float(np.nanmean(etc_rate)), root_depth_use=root_depth_use)
        mad_mm = np.maximum(taw - raw, 0.0).astype(np.float32)
        ks = compute_stress_coefficient(st_prev, mad_mm=mad_mm, wp_mm=np.zeros_like(taw, dtype=np.float32))
        eta = (ks * etc_term).astype(np.float32)

        if fmax_mm_day is None:
            fmax_source = self.fmax_layer if fmax_layer is None else fmax_layer
            if fmax_source is None:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "fmax_mm_day and fmax_layer both None; using USDA class lookup as fallback."
                )
                fmax_term = self._default_fmax_mm_day_from_soil() * float(dt_days)
            else:
                fmax_layer_arr = self._as_grid_array(fmax_source, name="fmax_layer")
                fmax_term = (np.maximum(fmax_layer_arr, 0.0) * float(dt_days)).astype(np.float32)
        else:
            fmax_term = (
                np.maximum(self._as_grid_array(fmax_mm_day, name="fmax_mm_day"), 0.0)
                * float(dt_days)
            ).astype(np.float32)

        d_term = compute_watneeds_deep_percolation(
            st_prev_mm=st_prev,
            mad_mm=mad_mm,
            taw_mm=np.maximum(taw, 0.0),
            fmax_mm=fmax_term,
        )

        available = st_prev + peff_term + irrigation_term
        total_losses = eta + d_term
        scale = np.ones_like(available, dtype=np.float32)
        neg_mask = (available - total_losses) < 0.0
        valid_losses = neg_mask & (total_losses > 0.0)
        scale[valid_losses] = available[valid_losses] / total_losses[valid_losses]
        scale = np.clip(scale, 0.0, 1.0)
        eta = eta * scale
        d_term = d_term * scale

        balance_after_d = available - eta - d_term
        r_term = np.maximum(balance_after_d - np.maximum(taw, 0.0), 0.0).astype(np.float32)
        st_next = np.clip(balance_after_d - r_term, 0.0, np.maximum(taw, 0.0)).astype(np.float32)
        total_runoff = (surface_runoff_term + r_term).astype(np.float32)

        nodata = self.soil.hydraulic_nodata
        for arr in (
            peff_term,
            irrigation_term,
            etc_term,
            eta,
            d_term,
            r_term,
            surface_runoff_term,
            total_runoff,
            st_next,
            fmax_term,
            mad_mm,
            taw,
            raw,
            ks,
        ):
            arr[~valid_mask] = nodata

        return {
            "soil_moisture_next_mm": st_next,
            "effective_precipitation_mm": peff_term,
            "irrigation_applied_mm": irrigation_term,
            "etc_mm": etc_term,
            "eta_mm": eta,
            "ks": ks,
            "d_t_mm": d_term,
            "r_t_mm": r_term,
            "overflow_runoff_mm": r_term,
            "surface_runoff_mm": surface_runoff_term,
            "total_runoff_mm": total_runoff,
            "fmax_mm": fmax_term,
            "raw_depletion_mm": raw,
            "stress_threshold_mm": mad_mm,
            "taw_mm": taw,
        }



    def summary(self) -> dict:
        return {
            "time_range": self.time_range,
            "land_cover_path": self.crop_fractions.path,
            "land_cover_bands": len(self.crop_fractions.crop_type),
            "crop_type": self.crop_fractions.crop_type,
            "has_crop_parameters": self.crop_fractions.crop_parameters is not None,
            "soil_path": self.soil.path,
            "soil_shape": list(self.soil.soil_type.shape),
            "has_taw_layer": self.soil.taw_mm is not None,
            "has_smax_layer": self.soil.smax_mm is not None,
            "has_fmax_layer": self.fmax_layer is not None,
            "soil_field_capacity_shape": list(self.soil.field_capacity.shape),
            "soil_wilting_point_shape": list(self.soil.wilting_point.shape),
            "soil_hydraulic_nodata": self.soil.hydraulic_nodata,
            "initial_condition": self.initial_condition,
            "has_initial_saturation": self.initial_saturation is not None,
            "peff_reduction_pct": self.peff_reduction_pct,
        }