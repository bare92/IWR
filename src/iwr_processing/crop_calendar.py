"""Crop calendar and crop coefficient (Kc) time-series management.

Issue H: Crop calendars and Kc curves
=====================================

WATNEEDS operates on full growing seasons with daily timesteps, requiring:
- Planting and harvest dates (day-of-year or absolute dates)
- Growth-stage fractions (initial, development, mid-season, late-season)
- Time-varying crop coefficients (Kc) per stage
- Off-season Kc = 0.5 (default)
- Aggregation to monthly/yearly green and blue water outputs

This module provides placeholders and data structures for future implementation
once datasource(s) are finalized (FAO-56, MIRCA, Siebert-Döll, or local data).

ROADMAP:
--------
1. CropCalendar dataclass: Stores planting/harvest dates and Kc schedule per crop.
2. compute_kc_daily(): Returns Kc for a given date within a growing season.
3. TimeSeriesDriver: Orchestrates daily water balance over full year with spin-up.
4. Aggregators: Functions to sum daily green/blue water to monthly/seasonal/yearly.

Note: Only data structures and method signatures are provided here. Actual
implementations depend on finalized datasource format and spatial coverage.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class GrowthStage(Enum):
    """FAO-56 crop growth stages."""
    INITIAL = "initial"  # 0-10% season, Kc = Kc_ini
    DEVELOPMENT = "development"  # 10-50% season, Kc rises linearly
    MID_SEASON = "mid_season"  # 50-80% season, Kc = Kc_mid (peak)
    LATE_SEASON = "late_season"  # 80-100% season, Kc falls linearly
    OFF_SEASON = "off_season"  # Outside growing season


@dataclass
class CropGrowthSchedule:
    """FAO-56-style Kc schedule for one crop variety.
    
    Attributes:
        crop_id: Unique crop identifier (matches CropParameter.crop_id).
        kc_ini: Crop coefficient during initial growth stage.
        kc_mid: Crop coefficient at mid-season (maximum).
        kc_end: Crop coefficient at end of season (harvest).
        kc_off_season: Crop coefficient during off-season (default 0.5).
        growth_stage_days: dict mapping GrowthStage to duration in days.
            Example: {GrowthStage.INITIAL: 20, GrowthStage.DEVELOPMENT: 40, ...}
    """
    crop_id: str
    kc_ini: float
    kc_mid: float
    kc_end: float
    kc_off_season: float = 0.5
    growth_stage_days: Optional[dict[GrowthStage, int]] = None

    def get_stage_duration_days(self, stage: GrowthStage) -> int:
        """Return duration of growth stage in days."""
        if self.growth_stage_days is None:
            raise ValueError(f"No growth_stage_days defined for crop {self.crop_id}")
        if stage not in self.growth_stage_days:
            raise ValueError(f"Stage {stage} not in growth_stage_days for crop {self.crop_id}")
        return self.growth_stage_days[stage]

    @property
    def total_season_days(self) -> int:
        """Total length of growing season in days."""
        if self.growth_stage_days is None:
            return 0
        return sum(self.growth_stage_days.values())

    def validate_total_days(self, expected_days: int) -> None:
        """Validate that stage durations match season length."""
        total_days = self.total_season_days
        if total_days != expected_days:
            raise ValueError(
                f"Growth-stage durations for crop {self.crop_id} sum to {total_days} days, "
                f"but planting/harvest imply {expected_days} days."
            )


@dataclass
class CropCalendar:
    """Planting and harvest dates for one crop in one location/year.
    
    Attributes:
        crop_id: Crop identifier (matches CropParameter.crop_id).
        year: Calendar year.
        planting_doy: Day of year for planting (1-366).
        harvest_doy: Day of year for harvest (1-366).
            If harvest_doy <= planting_doy, assumes harvest in next year.
        growth_schedule: CropGrowthSchedule with Kc values and stage durations.
    """
    crop_id: str
    year: int
    planting_doy: int
    harvest_doy: int
    growth_schedule: CropGrowthSchedule

    def __post_init__(self):
        """Validate DOY ranges."""
        if not (1 <= self.planting_doy <= 366):
            raise ValueError(f"planting_doy must be in [1, 366], got {self.planting_doy}")
        if not (1 <= self.harvest_doy <= 366):
            raise ValueError(f"harvest_doy must be in [1, 366], got {self.harvest_doy}")
        if self.crop_id != self.growth_schedule.crop_id:
            raise ValueError(
                f"Crop calendar crop_id {self.crop_id} does not match growth schedule "
                f"crop_id {self.growth_schedule.crop_id}."
            )
        self.growth_schedule.validate_total_days(self.season_length_days)

    @property
    def crosses_year_boundary(self) -> bool:
        """True if harvest is in next calendar year."""
        return self.harvest_doy <= self.planting_doy

    @property
    def season_length_days(self) -> int:
        """Inclusive season length implied by planting and harvest dates."""
        if self.crosses_year_boundary:
            days_in_year = 366 if self._is_leap_year(self.year) else 365
            return (days_in_year - self.planting_doy + 1) + self.harvest_doy
        return self.harvest_doy - self.planting_doy + 1

    def is_growing_season(self, doy: int, year: int) -> bool:
        """Check if a given date (year, doy) is within growing season."""
        if not (1 <= doy <= 366):
            raise ValueError(f"doy must be in [1, 366], got {doy}")

        if self.crosses_year_boundary:
            # Crop spans two calendar years
            if year == self.year:
                return doy >= self.planting_doy
            elif year == self.year + 1:
                return doy <= self.harvest_doy
            else:
                return False
        else:
            # Crop within single calendar year
            return year == self.year and self.planting_doy <= doy <= self.harvest_doy

    def days_since_planting(self, doy: int, year: int) -> Optional[int]:
        """Return days elapsed since planting, or None if not in season."""
        if not self.is_growing_season(doy, year):
            return None

        if self.crosses_year_boundary:
            if year == self.year:
                # In year of planting; count from planting_doy
                return doy - self.planting_doy
            else:
                # In year after planting; days from end of prev year + days in current year
                days_in_prev_year = 366 if self._is_leap_year(self.year) else 365
                return (days_in_prev_year - self.planting_doy) + doy
        else:
            # Single-year crop
            return doy - self.planting_doy

    @staticmethod
    def _is_leap_year(year: int) -> bool:
        """Check if year is a leap year."""
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def compute_kc_daily(
    calendar: CropCalendar,
    doy: int,
    year: int,
) -> float:
    """Compute crop coefficient (Kc) for a given date.
    
    Returns static Kc value based on crop growth phase (no interpolation).
    Returns Kc_off_season if outside growing season.
    
    Args:
        calendar: CropCalendar with planting/harvest and growth schedule.
        doy: Day of year (1-366).
        year: Calendar year.
    
    Returns:
        Crop coefficient value corresponding to current growth phase.
    
    """
    if not calendar.is_growing_season(doy, year):
        return calendar.growth_schedule.kc_off_season

    days_since_planting = calendar.days_since_planting(doy, year)
    if days_since_planting is None:
        return calendar.growth_schedule.kc_off_season

    schedule = calendar.growth_schedule
    days_initial = schedule.get_stage_duration_days(GrowthStage.INITIAL)
    days_development = schedule.get_stage_duration_days(GrowthStage.DEVELOPMENT)
    days_mid = schedule.get_stage_duration_days(GrowthStage.MID_SEASON)

    if days_since_planting < days_initial:
        return float(schedule.kc_ini)

    if days_since_planting < days_initial + days_development:
        return float(schedule.kc_ini)

    if days_since_planting < days_initial + days_development + days_mid:
        return float(schedule.kc_mid)

    return float(schedule.kc_end)


def load_crop_calendars_from_csv(
    csv_path: str,
    years: list[int] | tuple[int, ...] | None = None,
) -> dict[tuple[str, int], CropCalendar]:
    """Load crop calendars from CSV (placeholder).
    
    Expected columns:
        crop_id, [year], planting_doy, harvest_doy, kc_ini, kc_mid, kc_end,
        kc_off_season, [growth_stage_days_initial, growth_stage_days_development, ...]
    
    Returns:
        Dict keyed by (crop_id, year) for quick lookup.
    
    Required columns:
        crop_id, planting_doy, harvest_doy, kc_ini, kc_mid, kc_end,
        growth_stage_days_initial, growth_stage_days_development,
        growth_stage_days_mid_season, growth_stage_days_late_season
        If 'year' is missing, caller must pass `years` and each row is expanded
        for all requested years.
    Optional columns:
        kc_off_season
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Crop calendar CSV not found: {csv_path}")

    df = pd.read_csv(csv_file)
    required_columns = {
        "crop_id",
        "planting_doy",
        "harvest_doy",
        "kc_ini",
        "kc_mid",
        "kc_end",
        "growth_stage_days_initial",
        "growth_stage_days_development",
        "growth_stage_days_mid_season",
        "growth_stage_days_late_season",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Crop calendar CSV is missing columns: {sorted(missing_columns)}"
        )

    has_year_column = "year" in df.columns
    if not has_year_column and not years:
        raise ValueError(
            "Crop calendar CSV has no 'year' column. "
            "Provide `years=[...]` to load a year-less template."
        )

    calendars: dict[tuple[str, int], CropCalendar] = {}
    for _, row in df.iterrows():
        crop_id = str(row["crop_id"]).strip()
        if pd.isna(row["planting_doy"]) or pd.isna(row["harvest_doy"]):
            raise ValueError(
                f"Crop '{crop_id}' has empty planting_doy/harvest_doy. "
                "Fill calendar dates before running the model."
            )

        row_years: list[int]
        if has_year_column:
            row_years = [int(row["year"])]
        else:
            row_years = [int(y) for y in (years or [])]

        schedule = CropGrowthSchedule(
            crop_id=crop_id,
            kc_ini=float(row["kc_ini"]),
            kc_mid=float(row["kc_mid"]),
            kc_end=float(row["kc_end"]),
            kc_off_season=float(row["kc_off_season"]) if "kc_off_season" in df.columns else 0.5,
            growth_stage_days={
                GrowthStage.INITIAL: int(row["growth_stage_days_initial"]),
                GrowthStage.DEVELOPMENT: int(row["growth_stage_days_development"]),
                GrowthStage.MID_SEASON: int(row["growth_stage_days_mid_season"]),
                GrowthStage.LATE_SEASON: int(row["growth_stage_days_late_season"]),
            },
        )
        for year in row_years:
            calendar = CropCalendar(
                crop_id=crop_id,
                year=year,
                planting_doy=int(row["planting_doy"]),
                harvest_doy=int(row["harvest_doy"]),
                growth_schedule=schedule,
            )
            key = (crop_id, year)
            if key in calendars:
                raise ValueError(f"Duplicate crop calendar entry for {key}.")
            calendars[key] = calendar

    return calendars


class TimeSeriesDriver:
    """Orchestrates daily water-balance simulation over full year(s) with spin-up.
    
    Purpose:
        Ties together crop calendars, daily forcing data, soil initialization,
        and the IWRModel.watneeds_soil_water_balance_step() kernel.
    
    Workflow (sketch):
        1. Load crop calendars and growth schedules
        2. Run spin-up period (3 years) to equilibrate soil moisture
        3. Loop over analysis period (daily):
           a. Get Kc from crop calendar
           b. Compute ETc = Kc * ET0
           c. Call watneeds_soil_water_balance_step()
           d. Accumulate outputs
        4. Aggregate daily → monthly → seasonal → yearly
    
    Note:
        This class is a structural placeholder. Implementation details depend on:
        - Spatial handling (per-pixel calendars? or gridded calendar rasters?)
        - Temporal resolution of forcing data (daily, 3-daily, etc.)
        - Multi-crop handling (per-pixel fractions vs. homogeneous cells)
        - Memory/I/O strategy for large rasters and long time series
    """

    def __init__(
        self,
        iwr_model,  # IWRModel instance
        crop_calendars: dict,  # {(crop_id, year): CropCalendar, ...}
        start_date,  # date or similar
        end_date,
        precipitation_by_date: dict[date, Any],
        et0_by_date: dict[date, Any],
        etc_by_date: dict[date, Any] | None = None,
        spinup_years: int = 3,
        irrigated_mask: np.ndarray | str | None = None,
        weight_by_crop_fraction: bool = True,
    ):
        """Initialize time-series driver.
        
        Args:
            iwr_model: Initialized IWRModel instance.
            crop_calendars: Dict of CropCalendar instances.
            start_date: Start of analysis period.
            end_date: End of analysis period.
            precipitation_by_date: Mapping from date to daily precipitation.
            et0_by_date: Mapping from date to daily ET0.
            etc_by_date: Optional mapping from date to direct daily ETc.
            spinup_years: Years to spin-up before analysis period.
        """
        self.iwr_model = iwr_model
        self.crop_calendars = crop_calendars
        self.start_date = start_date
        self.end_date = end_date
        self.precipitation_by_date = precipitation_by_date
        self.et0_by_date = et0_by_date
        self.etc_by_date = etc_by_date
        self.spinup_years = spinup_years
        self.irrigated_mask = irrigated_mask
        self.weight_by_crop_fraction = weight_by_crop_fraction
        self._daily_outputs: list[dict[str, Any]] = []
        self._monthly_aggregates: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
        self._seasonal_aggregates: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._yearly_aggregates: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def _date_range(self) -> list[date]:
        current = self.start_date
        dates: list[date] = []
        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def _calendar_for_date(self, crop: str, current_date: date) -> CropCalendar | None:
        direct_key = (crop, current_date.year)
        previous_key = (crop, current_date.year - 1)
        for key in (direct_key, previous_key):
            calendar = self.crop_calendars.get(key)
            if calendar is not None and calendar.is_growing_season(
                current_date.timetuple().tm_yday,
                current_date.year,
            ):
                return calendar
        return None

    def _weight_output(self, crop: str, value: np.ndarray) -> np.ndarray:
        if not self.weight_by_crop_fraction:
            return value.astype(np.float32)
        fraction = self.iwr_model.crop_fractions.get_crop_fraction(crop).astype(np.float32)
        return (value * (fraction / 100.0)).astype(np.float32)

    def _forcing_value(self, mapping: dict[date, Any], current_date: date, name: str) -> Any:
        if current_date not in mapping:
            raise KeyError(f"Missing {name} forcing for date {current_date.isoformat()}.")
        return mapping[current_date]

    def run(self):
        """Execute daily WATNEEDS simulation with optional spin-up and aggregation."""
        self._daily_outputs = []
        self._monthly_aggregates = {}
        self._seasonal_aggregates = {}
        self._yearly_aggregates = {}

        dates = self._date_range()
        if not dates:
            return []

        for crop in self.iwr_model.crop_fractions.crop_type:
            crop_dates = [d for d in dates if self._calendar_for_date(crop, d) is not None]
            if not crop_dates:
                continue

            precip_series: list[Any] = []
            et0_series: list[Any] = []
            etc_series: list[Any] = []
            kc_series: list[float] = []
            for current_date in crop_dates:
                calendar = self._calendar_for_date(crop, current_date)
                if calendar is None:
                    continue
                doy = current_date.timetuple().tm_yday
                kc_series.append(compute_kc_daily(calendar, doy, current_date.year))
                precip_series.append(self._forcing_value(self.precipitation_by_date, current_date, "precipitation"))
                et0_series.append(self._forcing_value(self.et0_by_date, current_date, "et0"))
                if self.etc_by_date is not None:
                    etc_series.append(self._forcing_value(self.etc_by_date, current_date, "etc"))

            state = self.iwr_model.get_crop_initial_soil_water_mm(crop)
            if self.spinup_years > 0 and precip_series:
                spinup_kwargs: dict[str, Any] = {
                    "crop": crop,
                    "precipitation_mm_day_series": precip_series,
                    "is_irrigated": True,
                    "irrigation_mask": self.irrigated_mask,
                    "spinup_years": self.spinup_years,
                }
                if self.etc_by_date is not None:
                    spinup_kwargs["etc_mm_day_series"] = etc_series
                else:
                    spinup_kwargs["et0_mm_day_series"] = et0_series
                    spinup_kwargs["kc_series"] = kc_series
                spinup = self.iwr_model.watneeds_spinup(**spinup_kwargs)
                state = spinup["soil_moisture_spinup_end_mm"]

            for idx, current_date in enumerate(crop_dates):
                calendar = self._calendar_for_date(crop, current_date)
                if calendar is None:
                    continue
                doy = current_date.timetuple().tm_yday
                kc = compute_kc_daily(calendar, doy, current_date.year)
                step_kwargs: dict[str, Any] = {
                    "crop": crop,
                    "s_prev_mm": state,
                    "precipitation_mm": precip_series[idx],
                    "irrigated_mask": self.irrigated_mask,
                    "peff_coeff": 1.0 - (self.iwr_model.peff_reduction_pct / 100.0),
                }
                if self.etc_by_date is not None:
                    step_kwargs["etc_mm"] = etc_series[idx]
                else:
                    et0_value = et0_series[idx]
                    kc_value = kc
                    etc_value = np.asarray(et0_value, dtype=np.float32) * np.float32(kc_value)
                    step_kwargs["etc_mm"] = etc_value
                step = self.iwr_model.green_water_step(**step_kwargs)
                state = step["s_next_mm"]

                green_weighted = self._weight_output(crop, step["green_et_mm"])
                blue_weighted = self._weight_output(crop, step["blue_water_mm"])
                record = {
                    "crop": crop,
                    "date": current_date,
                    "kc": kc,
                    "green_water_mm": green_weighted,
                    "blue_water_mm": blue_weighted,
                    "pure_green_water_mm": step["green_et_mm"],
                    "pure_blue_water_mm": step["blue_water_mm"],
                }
                self._daily_outputs.append(record)

                month_key = (crop, current_date.year, current_date.month)
                season_key = (crop, calendar.year)
                year_key = (crop, current_date.year)
                for store, key in (
                    (self._monthly_aggregates, month_key),
                    (self._seasonal_aggregates, season_key),
                    (self._yearly_aggregates, year_key),
                ):
                    if key not in store:
                        store[key] = {
                            "green_water_mm": np.zeros_like(green_weighted, dtype=np.float32),
                            "blue_water_mm": np.zeros_like(blue_weighted, dtype=np.float32),
                        }
                    store[key]["green_water_mm"] += green_weighted
                    store[key]["blue_water_mm"] += blue_weighted

        return self._daily_outputs

    def get_daily_outputs(self):
        """Return accumulated daily outputs."""
        return self._daily_outputs

    def get_monthly_aggregates(self):
        """Return monthly aggregated green/blue water."""
        return self._monthly_aggregates

    def get_seasonal_aggregates(self):
        """Return seasonal aggregates per crop."""
        return self._seasonal_aggregates

    def get_yearly_aggregates(self):
        """Return yearly aggregates per crop."""
        return self._yearly_aggregates


# Placeholder aggregation functions
def aggregate_daily_to_monthly(
    daily_values: dict,  # {date: value, ...}
) -> dict:
    """Aggregate daily values to monthly sums.
    
    Args:
        daily_values: Dict mapping date to daily value (mm or similar).
    
    Returns:
        Dict mapping (year, month) to monthly total.
    """
    monthly: dict[tuple[int, int], np.ndarray] = {}
    for current_date, value in daily_values.items():
        key = (current_date.year, current_date.month)
        array_value = np.asarray(value, dtype=np.float32)
        if key not in monthly:
            monthly[key] = np.zeros_like(array_value, dtype=np.float32)
        monthly[key] += array_value
    return monthly


def aggregate_daily_to_yearly(
    daily_values: dict,
) -> dict:
    """Aggregate daily values to annual sums.
    
    Args:
        daily_values: Dict mapping date to daily value (mm or similar).
    
    Returns:
        Dict mapping year to annual total.
    """
    yearly: dict[int, np.ndarray] = {}
    for current_date, value in daily_values.items():
        key = current_date.year
        array_value = np.asarray(value, dtype=np.float32)
        if key not in yearly:
            yearly[key] = np.zeros_like(array_value, dtype=np.float32)
        yearly[key] += array_value
    return yearly


def aggregate_by_crop_fraction(
    daily_values_per_pixel: np.ndarray,  # shape: (rows, cols, days)
    crop_fractions: np.ndarray,  # shape: (rows, cols) or (rows, cols, crops)
    crop_name: str,
) -> np.ndarray:
    """Aggregate daily pixel-level values by crop fraction.
    
    Args:
        daily_values_per_pixel: Daily values for each pixel.
        crop_fractions: Fractional area per crop per pixel.
        crop_name: Name of crop to aggregate.
    
    Returns:
        Aggregated values per pixel for the crop.
    """
    del crop_name
    daily_values = np.asarray(daily_values_per_pixel, dtype=np.float32)
    fractions = np.asarray(crop_fractions, dtype=np.float32)
    if fractions.ndim == 2:
        weights = fractions / 100.0
    elif fractions.ndim == 3 and fractions.shape[-1] == 1:
        weights = fractions[..., 0] / 100.0
    else:
        raise ValueError(
            "crop_fractions must be 2D or 3D with a single crop layer for aggregation."
        )
    if daily_values.ndim == 2:
        return (daily_values * weights).astype(np.float32)
    if daily_values.ndim == 3:
        return (daily_values * weights[:, :, np.newaxis]).astype(np.float32)
    raise ValueError("daily_values_per_pixel must be 2D or 3D.")
