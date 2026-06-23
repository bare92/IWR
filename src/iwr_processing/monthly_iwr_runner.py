"""Month-by-month WATNEEDS runner for streaming NetCDF forcing archives.

This module is meant for long simulations where forcing is stored as monthly
NetCDF files with a daily time dimension. It processes one month at a time,
keeps the last soil storage as the starting point for the next month, and can
write one NetCDF output per processed month.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from iwr_processing.crop_calendar import CropCalendar, compute_kc_daily
from iwr_processing.iwr_core_process import (
    IWRModel,
    PHENOLOGY_STAGE_GROWING,
    PHENOLOGY_STAGE_INACTIVE,
    PHENOLOGY_STAGE_MAXIMUM,
    PHENOLOGY_STAGE_NODATA,
    PHENOLOGY_STAGE_SENESCENCE,
    compute_kc_from_phenology_stage,
)
from iwr_processing.netcdf_forcing_reader import MonthlyForcingChunk, MonthlyNetCDFForcingReader


logger = logging.getLogger(__name__)


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
        phenology_layers: dict[str, np.ndarray] | None = None,
        phenology_nodata_value: int | float | None = None,
        use_phenology_kc: bool = False,
        skip_iwr_when_inactive: bool = False,
    ):
        self.model = model
        self.forcing_reader = forcing_reader
        self.crop_calendar = crop_calendar
        self.crop_name = crop_name
        self.output_dir = Path(output_dir)
        self.irrigated_mask = irrigated_mask
        self.use_direct_etc = use_direct_etc
        self.write_monthly_outputs = write_monthly_outputs
        self.phenology_layers = phenology_layers or {}
        self.phenology_nodata_value = phenology_nodata_value
        self.use_phenology_kc = bool(use_phenology_kc)
        self.skip_iwr_when_inactive = bool(skip_iwr_when_inactive)
        self._validate_phenology_configuration()

    def _validate_phenology_configuration(self) -> None:
        if not self.use_phenology_kc:
            return

        required_s1 = ("phenoe1", "phenom1", "phenos1", "phenosen1")
        missing_s1 = [name for name in required_s1 if name not in self.phenology_layers]
        if missing_s1:
            raise ValueError(
                "use_phenology_kc=True requires season-1 phenology layers "
                f"{required_s1}. Missing: {missing_s1}"
            )

        shape = np.asarray(self.phenology_layers[required_s1[0]]).shape
        for key, arr in self.phenology_layers.items():
            if np.asarray(arr).shape != shape:
                raise ValueError(
                    f"Phenology layer '{key}' shape {np.asarray(arr).shape} "
                    f"does not match expected shape {shape}."
                )

        required_s2 = ("phenoe2", "phenom2", "phenos2", "phenosen2")
        present_s2 = [name in self.phenology_layers for name in required_s2]
        if any(present_s2) and not all(present_s2):
            raise ValueError(
                "Season-2 phenology layers must be all provided or all omitted: "
                f"{required_s2}."
            )

    def _build_step_kwargs(
        self,
        current_date: date,
        chunk: MonthlyForcingChunk,
        day_index: int,
        soil_storage_mm: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        precipitation = chunk.data["precipitation"][day_index]
        diagnostics: dict[str, np.ndarray] = {}

        if self.use_direct_etc and "etc" in chunk.data:
            etc_mm = chunk.data["etc"][day_index]
        else:
            et0 = chunk.data["et0"][day_index]

            if self.use_phenology_kc:
                schedule = self.crop_calendar.growth_schedule
                doy = current_date.timetuple().tm_yday

                stage, kc = compute_kc_from_phenology_stage(
                    doy=doy,
                    sos_1=np.asarray(self.phenology_layers["phenoe1"], dtype=np.float32),
                    tom_1=np.asarray(self.phenology_layers["phenom1"], dtype=np.float32),
                    sen_1=np.asarray(self.phenology_layers["phenos1"], dtype=np.float32),
                    eos_1=np.asarray(self.phenology_layers["phenosen1"], dtype=np.float32),
                    sos_2=np.asarray(self.phenology_layers["phenoe2"], dtype=np.float32)
                    if "phenoe2" in self.phenology_layers
                    else None,
                    tom_2=np.asarray(self.phenology_layers["phenom2"], dtype=np.float32)
                    if "phenom2" in self.phenology_layers
                    else None,
                    sen_2=np.asarray(self.phenology_layers["phenos2"], dtype=np.float32)
                    if "phenos2" in self.phenology_layers
                    else None,
                    eos_2=np.asarray(self.phenology_layers["phenosen2"], dtype=np.float32)
                    if "phenosen2" in self.phenology_layers
                    else None,
                    kc_ini=float(schedule.kc_ini),
                    kc_mid=float(schedule.kc_mid),
                    kc_end=float(schedule.kc_end),
                    kc_off=float(schedule.kc_off_season),
                    nodata_value=self.phenology_nodata_value,
                )

                etc_mm = np.asarray(et0, dtype=np.float32) * np.asarray(kc, dtype=np.float32)
                if self.skip_iwr_when_inactive:
                    etc_mm = np.where(stage == PHENOLOGY_STAGE_INACTIVE, 0.0, etc_mm).astype(np.float32)

                diagnostics["phenology_stage"] = stage.astype(np.uint8)
                diagnostics["kc"] = np.asarray(kc, dtype=np.float32)
            else:
                kc = compute_kc_daily(
                    self.crop_calendar,
                    current_date.timetuple().tm_yday,
                    current_date.year,
                )
                etc_mm = np.asarray(et0, dtype=np.float32) * np.float32(kc)

        return {
            "crop": self.crop_name,
            "s_prev_mm": soil_storage_mm,
            "precipitation_mm": precipitation,
            "etc_mm": etc_mm,
            "irrigated_mask": self.irrigated_mask,
        }, diagnostics

    def _log_daily_phenology_counts(self, current_date: date, stage: np.ndarray) -> None:
        inactive = int(np.count_nonzero(stage == PHENOLOGY_STAGE_INACTIVE))
        growing = int(np.count_nonzero(stage == PHENOLOGY_STAGE_GROWING))
        maximum = int(np.count_nonzero(stage == PHENOLOGY_STAGE_MAXIMUM))
        senescence = int(np.count_nonzero(stage == PHENOLOGY_STAGE_SENESCENCE))
        nodata = int(np.count_nonzero(stage == PHENOLOGY_STAGE_NODATA))
        active = growing + maximum + senescence

        logger.info(
            "Phenology stage counts %s | active=%d inactive=%d growing=%d maximum=%d senescence=%d nodata=%d",
            current_date.isoformat(),
            active,
            inactive,
            growing,
            maximum,
            senescence,
            nodata,
        )

    @staticmethod
    def _stack_monthly_records(
        records: list[dict[str, np.ndarray]],
        key: str,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        return np.stack([record[key] for record in records], axis=0).astype(dtype)

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

        if "kc" in daily_records[0]:
            data_vars["kc"] = (("time", "y", "x"), self._stack_monthly_records(daily_records, "kc"))
        if "phenology_stage" in daily_records[0]:
            data_vars["phenology_stage"] = (
                ("time", "y", "x"),
                self._stack_monthly_records(daily_records, "phenology_stage", dtype=np.uint8),
            )

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
                step_kwargs, diagnostics = self._build_step_kwargs(current_date, chunk, day_index, soil_storage_mm)
                if "phenology_stage" in diagnostics:
                    self._log_daily_phenology_counts(current_date, diagnostics["phenology_stage"])
                step = self.model.green_water_step(**step_kwargs)
                step.update(diagnostics)
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
