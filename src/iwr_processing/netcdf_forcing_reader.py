"""Streaming forcing reader for monthly NetCDF files with daily timesteps.

This module is dedicated to the DAO-style layout where forcing variables are
stored in separate folders, one monthly NetCDF per variable, with a daily time
dimension inside each file.

Expected folder layout example:

    root/
      ET_HS/
        potential_evapotranspiration_2015_01.nc
      P/
        precipitation_2015_01.nc
      Temperature/
        temperature_2015_01.nc

The reader is streaming-friendly: it loads one month at a time and can yield
one day at a time without requiring the full multi-year archive in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterator

import numpy as np
import xarray as xr
import pandas as pd


@dataclass(frozen=True)
class MonthlyNetCDFVariableSpec:
    """Specification for one forcing variable stored as monthly NetCDF files."""

    name: str
    folder: str
    file_pattern: str = "{variable}_{year}_{month:02d}.nc"
    variable_name: str | None = None
    units: str | None = None


@dataclass(frozen=True)
class MonthlyForcingChunk:
    """One month of forcing data for one or more variables."""

    year: int
    month: int
    dates: list[date]
    data: dict[str, np.ndarray]


class MonthlyNetCDFForcingReader:
    """Read daily forcing arrays from a folder of monthly NetCDF files.

    The reader resolves the date from each NetCDF `time` coordinate and returns
    a model-grid numpy array for the requested day.

    Parameters
    ----------
    root_dir:
        Root directory containing one subfolder per variable.
    variable_specs:
        Mapping from logical forcing name to variable specification.
    start_date, end_date:
        Optional analysis window. If omitted, the reader can still be used via
        `get_day()` or `iter_daily()` over the available files.
    file_regex:
        Optional custom regex for extracting year/month from file names.
        Defaults to `YYYY_MM` detection.
    """

    def __init__(
        self,
        root_dir: str | Path,
        variable_specs: dict[str, MonthlyNetCDFVariableSpec],
        start_date: date | None = None,
        end_date: date | None = None,
        file_regex: str = r"(?P<year>\d{4})_(?P<month>\d{2})",
    ):
        self.root_dir = Path(root_dir)
        self.variable_specs = variable_specs
        self.start_date = start_date
        self.end_date = end_date
        self.file_regex = re.compile(file_regex)

    @classmethod
    def from_dao_layout(
        cls,
        root_dir: str | Path,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> "MonthlyNetCDFForcingReader":
        """Build a reader for the DAO folder layout used in this project."""

        specs = {
            "et0": MonthlyNetCDFVariableSpec(
                name="et0",
                folder="ET_HS",
                file_pattern="potential_evapotranspiration_{year}_{month:02d}.nc",
                variable_name="PET",
            ),
            "precipitation": MonthlyNetCDFVariableSpec(
                name="precipitation",
                folder="P",
                file_pattern="precipitation_{year}_{month:02d}.nc",
                variable_name=None,
            ),
            "temperature": MonthlyNetCDFVariableSpec(
                name="temperature",
                folder="Temperature",
                file_pattern="temperature_{year}_{month:02d}.nc",
                variable_name=None,
            ),
            "temperature_min": MonthlyNetCDFVariableSpec(
                name="temperature_min",
                folder="Temperature_min",
                file_pattern="temperature_min_{year}_{month:02d}.nc",
                variable_name=None,
            ),
            "temperature_max": MonthlyNetCDFVariableSpec(
                name="temperature_max",
                folder="Temperature_max",
                file_pattern="temperature_max_{year}_{month:02d}.nc",
                variable_name=None,
            ),
        }
        return cls(root_dir=root_dir, variable_specs=specs, start_date=start_date, end_date=end_date)

    def _month_path(self, spec: MonthlyNetCDFVariableSpec, year: int, month: int) -> Path:
        return self.root_dir / spec.folder / spec.file_pattern.format(
            variable=spec.name,
            year=year,
            month=month,
        )

    def _date_range(self) -> Iterator[date]:
        if self.start_date is None or self.end_date is None:
            raise ValueError("start_date and end_date are required for iter_daily().")

        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)

    @lru_cache(maxsize=24)
    def _load_month(self, variable: str, year: int, month: int) -> xr.Dataset:
        spec = self.variable_specs[variable]
        path = self._month_path(spec, year, month)
        if not path.exists():
            raise FileNotFoundError(f"Missing NetCDF forcing file: {path}")

        ds = xr.open_dataset(path, decode_times=True)
        if spec.variable_name is not None and spec.variable_name in ds:
            return ds[[spec.variable_name]]
        if spec.variable_name is None and len(ds.data_vars) == 1:
            return ds
        if variable in ds.data_vars:
            return ds[[variable]]
        if len(ds.data_vars) == 1:
            return ds
        raise KeyError(
            f"Cannot resolve variable for '{variable}' in {path}. Available vars: {list(ds.data_vars)}"
        )

    @staticmethod
    def _extract_array(ds: xr.Dataset, variable_name: str, target_date: date) -> np.ndarray:
        if "time" not in ds.coords:
            raise ValueError("NetCDF forcing dataset must contain a 'time' coordinate.")

        timestamp = np.datetime64(target_date)
        selection = ds[variable_name].sel(time=timestamp)
        if selection.ndim == 3:
            selection = selection.isel(time=0)
        return selection.values.astype(np.float32)

    def get_day(self, current_date: date) -> dict[str, np.ndarray]:
        """Return one forcing snapshot for the requested day."""
        if self.start_date is not None and current_date < self.start_date:
            raise ValueError(f"Date {current_date} is before start_date {self.start_date}.")
        if self.end_date is not None and current_date > self.end_date:
            raise ValueError(f"Date {current_date} is after end_date {self.end_date}.")

        month_values: dict[str, np.ndarray] = {}
        for variable, spec in self.variable_specs.items():
            ds = self._load_month(variable, current_date.year, current_date.month)
            data_var = spec.variable_name or variable
            if data_var not in ds.data_vars:
                if len(ds.data_vars) == 1:
                    data_var = next(iter(ds.data_vars))
                else:
                    raise KeyError(
                        f"Variable '{data_var}' not found in dataset for {variable}."
                    )
            month_values[variable] = self._extract_array(ds, data_var, current_date)
        return month_values

    def read_month(self, year: int, month: int) -> MonthlyForcingChunk:
        """Load one entire month of forcing arrays.

        Returns arrays with shape (time, rows, cols) for each configured variable.
        The time coordinate is preserved via the returned `dates` list.
        """
        month_start = date(year, month, 1)
        next_month = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
        window_start = max(month_start, self.start_date) if self.start_date is not None else month_start
        window_end = min(next_month - timedelta(days=1), self.end_date) if self.end_date is not None else next_month - timedelta(days=1)

        if window_start > window_end:
            raise ValueError(f"No dates available for {year}-{month:02d} within the configured window.")

        month_data: dict[str, np.ndarray] = {}
        dates: list[date] | None = None
        for variable, spec in self.variable_specs.items():
            ds = self._load_month(variable, year, month)
            data_var = spec.variable_name or variable
            if data_var not in ds.data_vars:
                if len(ds.data_vars) == 1:
                    data_var = next(iter(ds.data_vars))
                else:
                    raise KeyError(
                        f"Variable '{data_var}' not found in dataset for {variable}."
                    )
            selected = ds[data_var].sel(time=slice(np.datetime64(window_start), np.datetime64(window_end)))
            if dates is None:
                dates = [pd.Timestamp(value).date() for value in selected["time"].values]
            month_data[variable] = selected.values.astype(np.float32)

        if dates is None or not dates:
            raise ValueError(f"No dates available for {year}-{month:02d} within the configured window.")

        return MonthlyForcingChunk(year=year, month=month, dates=dates, data=month_data)

    def iter_months(self) -> Iterator[MonthlyForcingChunk]:
        """Iterate over month-sized forcing chunks within the configured window."""
        if self.start_date is None or self.end_date is None:
            raise ValueError("start_date and end_date are required for iter_months().")

        year = self.start_date.year
        month = self.start_date.month
        while True:
            current_month_start = date(year, month, 1)
            if current_month_start > self.end_date:
                break
            if current_month_start >= self.start_date or date(year, month, 1) == self.start_date.replace(day=1):
                yield self.read_month(year, month)

            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

    def iter_daily(self) -> Iterator[tuple[date, dict[str, np.ndarray]]]:
        """Iterate day by day over the configured analysis window."""
        for current_date in self._date_range():
            yield current_date, self.get_day(current_date)

    def available_months(self) -> list[tuple[int, int, str]]:
        """List months for which all configured variables have a file."""
        months: list[tuple[int, int, str]] = []
        for variable, spec in self.variable_specs.items():
            folder = self.root_dir / spec.folder
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.nc")):
                match = self.file_regex.search(path.name)
                if match:
                    months.append((int(match.group("year")), int(match.group("month")), variable))
        return months
