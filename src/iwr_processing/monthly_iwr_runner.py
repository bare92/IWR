"""Month-by-month WATNEEDS runner for streaming NetCDF forcing archives.

This module is meant for long simulations where forcing is stored as monthly
NetCDF files with a daily time dimension. It processes one month at a time,
keeps the last soil storage as the starting point for the next month, and can
write one NetCDF output per processed month.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from iwr_processing.crop_calendar import CropCalendar, compute_kc_daily
from iwr_processing.iwr_core_process import IWRModel
from iwr_processing.netcdf_forcing_reader import MonthlyForcingChunk, MonthlyNetCDFForcingReader


@dataclass(frozen=True)
class MonthlyIWRRunResult:
    """Summary of a streamed month-by-month run."""

    written_files: list[str]
    final_soil_storage_mm: np.ndarray


class MonthlyIWRRunner:
    """Process forcing data month by month while carrying soil storage forward."""

    def __init__(
        self,
        model: IWRModel,
        forcing_reader: MonthlyNetCDFForcingReader,
        crop_calendar: CropCalendar,
        crop_name: str,
        output_dir: str | Path,
        irrigated_mask: np.ndarray | str | None = None,
        use_direct_etc: bool = False,
        write_monthly_outputs: bool = True,
    ):
        self.model = model
        self.forcing_reader = forcing_reader
        self.crop_calendar = crop_calendar
        self.crop_name = crop_name
        self.output_dir = Path(output_dir)
        self.irrigated_mask = irrigated_mask
        self.use_direct_etc = use_direct_etc
        self.write_monthly_outputs = write_monthly_outputs

    def _build_step_kwargs(
        self,
        current_date: date,
        chunk: MonthlyForcingChunk,
        day_index: int,
        soil_storage_mm: np.ndarray,
    ) -> dict[str, Any]:
        precipitation = chunk.data["precipitation"][day_index]
        if self.use_direct_etc and "etc" in chunk.data:
            etc_mm = chunk.data["etc"][day_index]
        else:
            et0 = chunk.data["et0"][day_index]
            kc = compute_kc_daily(self.crop_calendar, current_date.timetuple().tm_yday, current_date.year)
            etc_mm = np.asarray(et0, dtype=np.float32) * np.float32(kc)

        return {
            "crop": self.crop_name,
            "s_prev_mm": soil_storage_mm,
            "precipitation_mm": precipitation,
            "etc_mm": etc_mm,
            "irrigated_mask": self.irrigated_mask,
        }

    @staticmethod
    def _stack_monthly_records(records: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
        return np.stack([record[key] for record in records], axis=0).astype(np.float32)

    def _write_monthly_output(
        self,
        chunk: MonthlyForcingChunk,
        daily_records: list[dict[str, np.ndarray]],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        time_index = np.array(chunk.dates, dtype="datetime64[ns]")
        data_vars: dict[str, Any] = {
            "soil_moisture_next_mm": (("time", "y", "x"), self._stack_monthly_records(daily_records, "s_next_mm")),
            "green_et_mm": (("time", "y", "x"), self._stack_monthly_records(daily_records, "green_et_mm")),
            "blue_water_mm": (("time", "y", "x"), self._stack_monthly_records(daily_records, "blue_water_mm")),
            "deep_perc_mm": (("time", "y", "x"), self._stack_monthly_records(daily_records, "deep_perc_mm")),
            "surface_runoff_mm": (("time", "y", "x"), self._stack_monthly_records(daily_records, "surface_runoff_mm")),
            "overflow_runoff_mm": (("time", "y", "x"), self._stack_monthly_records(daily_records, "overflow_runoff_mm")),
            "total_runoff_mm": (("time", "y", "x"), self._stack_monthly_records(daily_records, "total_runoff_mm")),
            "ks": (("time", "y", "x"), self._stack_monthly_records(daily_records, "ks")),
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": time_index, "y": np.arange(daily_records[0]["s_next_mm"].shape[0]), "x": np.arange(daily_records[0]["s_next_mm"].shape[1])},
            attrs={
                "crop_name": self.crop_name,
                "year": chunk.year,
                "month": chunk.month,
            },
        )
        ds.to_netcdf(output_path)

    def run(self) -> MonthlyIWRRunResult:
        """Run the IWR model month by month and carry soil storage across months."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        soil_storage_mm = self.model.get_crop_initial_soil_water_mm(self.crop_name)
        written_files: list[str] = []

        for chunk in self.forcing_reader.iter_months():
            daily_records: list[dict[str, np.ndarray]] = []

            for day_index, current_date in enumerate(chunk.dates):
                step_kwargs = self._build_step_kwargs(current_date, chunk, day_index, soil_storage_mm)
                step = self.model.green_water_step(**step_kwargs)
                soil_storage_mm = step["s_next_mm"]
                daily_records.append(step)

            if self.write_monthly_outputs:
                output_path = self.output_dir / f"iwr_daily_{chunk.year}_{chunk.month:02d}.nc"
                self._write_monthly_output(chunk, daily_records, output_path)
                written_files.append(str(output_path))

        return MonthlyIWRRunResult(
            written_files=written_files,
            final_soil_storage_mm=soil_storage_mm,
        )
