from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from iwr_processing.crop_calendar import CropCalendar, CropGrowthSchedule, GrowthStage
from iwr_processing.monthly_iwr_runner import MonthlyIWRRunner
from iwr_processing.netcdf_forcing_reader import MonthlyNetCDFForcingReader


def _write_monthly_nc(path, var_name, values, start_date):
    time = pd.date_range(start=start_date, periods=values.shape[0], freq="D")
    ds = xr.Dataset(
        {var_name: (("time", "y", "x"), values.astype(np.float32))},
        coords={"time": time, "y": np.arange(values.shape[1]), "x": np.arange(values.shape[2])},
    )
    ds.to_netcdf(path)


def _build_calendar() -> CropCalendar:
    return CropCalendar(
        crop_id="wheat",
        year=2015,
        planting_doy=1,
        harvest_doy=31,
        growth_schedule=CropGrowthSchedule(
            crop_id="wheat",
            kc_ini=0.4,
            kc_mid=1.0,
            kc_end=0.6,
            growth_stage_days={
                GrowthStage.INITIAL: 10,
                GrowthStage.DEVELOPMENT: 10,
                GrowthStage.MID_SEASON: 10,
                GrowthStage.LATE_SEASON: 1,
            },
        ),
    )


class DummyCropFractions:
    crop_type = ["wheat"]

    def get_crop_fraction(self, crop: str) -> np.ndarray:
        assert crop == "wheat"
        return np.ones((2, 2), dtype=np.float32) * 100.0


class DummyModel:
    def __init__(self):
        self.crop_fractions = DummyCropFractions()
        self.peff_reduction_pct = 5.0

    def get_crop_initial_soil_water_mm(self, crop: str) -> np.ndarray:
        assert crop == "wheat"
        return np.ones((2, 2), dtype=np.float32) * 10.0

    def green_water_step(self, **kwargs):
        s_prev = np.asarray(kwargs["s_prev_mm"], dtype=np.float32)
        precipitation = np.asarray(kwargs["precipitation_mm"], dtype=np.float32)
        etc = np.asarray(kwargs["etc_mm"], dtype=np.float32)
        s_next = np.clip(s_prev + precipitation * 0.95 - etc * 0.5, 0.0, 100.0)
        green_et = np.minimum(etc * 0.5, etc)
        blue = np.maximum(etc - green_et, 0.0)
        zeros = np.zeros_like(s_next, dtype=np.float32)
        return {
            "s_next_mm": s_next,
            "green_et_mm": green_et.astype(np.float32),
            "blue_water_mm": blue.astype(np.float32),
            "deep_perc_mm": zeros,
            "surface_runoff_mm": zeros,
            "overflow_runoff_mm": zeros,
            "total_runoff_mm": zeros,
            "ks": np.ones_like(s_next, dtype=np.float32),
        }


def test_monthly_runner_carries_soil_storage_forward(tmp_path):
    root = tmp_path / "forcing"
    output = tmp_path / "out"
    for folder in ["ET_HS", "P", "Temperature", "Temperature_min", "Temperature_max"]:
        (root / folder).mkdir(parents=True, exist_ok=True)

    et0_jan = np.stack([np.full((2, 2), 2.0, dtype=np.float32)] * 3)
    p_jan = np.stack([np.full((2, 2), 0.0, dtype=np.float32)] * 3)
    et0_feb = np.stack([np.full((2, 2), 2.0, dtype=np.float32)] * 2)
    p_feb = np.stack([np.full((2, 2), 0.0, dtype=np.float32)] * 2)

    _write_monthly_nc(root / "ET_HS" / "potential_evapotranspiration_2015_01.nc", "PET", et0_jan, date(2015, 1, 1))
    _write_monthly_nc(root / "P" / "precipitation_2015_01.nc", "precipitation", p_jan, date(2015, 1, 1))
    _write_monthly_nc(root / "Temperature" / "temperature_2015_01.nc", "temperature", et0_jan, date(2015, 1, 1))
    _write_monthly_nc(root / "Temperature_min" / "temperature_min_2015_01.nc", "temperature_min", et0_jan, date(2015, 1, 1))
    _write_monthly_nc(root / "Temperature_max" / "temperature_max_2015_01.nc", "temperature_max", et0_jan, date(2015, 1, 1))

    _write_monthly_nc(root / "ET_HS" / "potential_evapotranspiration_2015_02.nc", "PET", et0_feb, date(2015, 2, 1))
    _write_monthly_nc(root / "P" / "precipitation_2015_02.nc", "precipitation", p_feb, date(2015, 2, 1))
    _write_monthly_nc(root / "Temperature" / "temperature_2015_02.nc", "temperature", et0_feb, date(2015, 2, 1))
    _write_monthly_nc(root / "Temperature_min" / "temperature_min_2015_02.nc", "temperature_min", et0_feb, date(2015, 2, 1))
    _write_monthly_nc(root / "Temperature_max" / "temperature_max_2015_02.nc", "temperature_max", et0_feb, date(2015, 2, 1))

    model = DummyModel()

    reader = MonthlyNetCDFForcingReader.from_dao_layout(root, start_date=date(2015, 1, 1), end_date=date(2015, 2, 2))
    calendar = _build_calendar()

    runner = MonthlyIWRRunner(
        model=model,
        forcing_reader=reader,
        crop_calendar=calendar,
        crop_name="wheat",
        output_dir=output,
        write_monthly_outputs=True,
    )

    result = runner.run()

    assert len(result.written_files) == 2
    feb_output = xr.open_dataset(result.written_files[1])
    assert feb_output.sizes["time"] == 2
    assert np.all(np.isfinite(feb_output["soil_moisture_next_mm"].values))
    assert result.final_soil_storage_mm.shape == (2, 2)