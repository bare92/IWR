from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from iwr_processing.crop_calendar import CropCalendar, CropGrowthSchedule, GrowthStage
from iwr_processing.iwr_core_process import (
    PHENOLOGY_STAGE_INACTIVE,
    PHENOLOGY_STAGE_MAXIMUM,
    PHENOLOGY_STAGE_NODATA,
    PHENOLOGY_STAGE_SENESCENCE,
    classify_phenology_stage,
    compute_kc_from_phenology_stage,
    doy_to_dekad,
)
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


def test_doy_to_dekad_boundaries():
    assert doy_to_dekad(1) == 1
    assert doy_to_dekad(10) == 1
    assert doy_to_dekad(11) == 2
    assert doy_to_dekad(365) == 36
    assert doy_to_dekad(366) == 36


def test_classify_phenology_stage_cross_year_mapping():
    # DOY 100 -> dekad 10. Candidates: 10, 46, 82.
    # Cross-year season (30..47) should map to t=46.
    sos = np.array([[30]], dtype=np.float32)
    tom = np.array([[35]], dtype=np.float32)
    sen = np.array([[40]], dtype=np.float32)
    eos = np.array([[47]], dtype=np.float32)

    stage = classify_phenology_stage(100, sos=sos, tom=tom, sen=sen, eos=eos)
    assert stage.dtype == np.uint8
    assert stage[0, 0] == PHENOLOGY_STAGE_SENESCENCE


def test_compute_kc_from_phenology_stage_two_season_priority():
    # On DOY 100 (dekad 10), season-1 is senescence and season-2 is maximum.
    # Priority rule should pick maximum.
    shape = (1, 1)
    sos_1 = np.full(shape, 8, dtype=np.float32)
    tom_1 = np.full(shape, 9, dtype=np.float32)
    sen_1 = np.full(shape, 10, dtype=np.float32)
    eos_1 = np.full(shape, 11, dtype=np.float32)

    sos_2 = np.full(shape, 9, dtype=np.float32)
    tom_2 = np.full(shape, 10, dtype=np.float32)
    sen_2 = np.full(shape, 11, dtype=np.float32)
    eos_2 = np.full(shape, 12, dtype=np.float32)

    stage, kc = compute_kc_from_phenology_stage(
        doy=100,
        sos_1=sos_1,
        tom_1=tom_1,
        sen_1=sen_1,
        eos_1=eos_1,
        sos_2=sos_2,
        tom_2=tom_2,
        sen_2=sen_2,
        eos_2=eos_2,
        kc_ini=0.4,
        kc_mid=1.0,
        kc_end=0.6,
        kc_off=0.5,
    )

    assert stage[0, 0] == PHENOLOGY_STAGE_MAXIMUM
    assert np.isclose(kc[0, 0], 1.0)


def test_compute_kc_handles_nodata():
    nodata = -9999.0
    sos = np.array([[nodata]], dtype=np.float32)
    tom = np.array([[nodata]], dtype=np.float32)
    sen = np.array([[nodata]], dtype=np.float32)
    eos = np.array([[nodata]], dtype=np.float32)

    stage, kc = compute_kc_from_phenology_stage(
        doy=120,
        sos_1=sos,
        tom_1=tom,
        sen_1=sen,
        eos_1=eos,
        kc_ini=0.4,
        kc_mid=1.0,
        kc_end=0.6,
        kc_off=0.5,
        nodata_value=nodata,
    )

    assert stage[0, 0] == PHENOLOGY_STAGE_NODATA
    assert np.isnan(kc[0, 0])


def test_monthly_runner_with_phenology_outputs(tmp_path):
    root = tmp_path / "forcing"
    output = tmp_path / "out"
    for folder in ["ET_HS", "P", "Temperature", "Temperature_min", "Temperature_max"]:
        (root / folder).mkdir(parents=True, exist_ok=True)

    day = date(2015, 4, 10)  # DOY 100 -> dekad 10
    et0 = np.stack([np.full((2, 2), 2.0, dtype=np.float32)])
    p = np.stack([np.zeros((2, 2), dtype=np.float32)])

    _write_monthly_nc(root / "ET_HS" / "potential_evapotranspiration_2015_04.nc", "PET", et0, day)
    _write_monthly_nc(root / "P" / "precipitation_2015_04.nc", "precipitation", p, day)
    _write_monthly_nc(root / "Temperature" / "temperature_2015_04.nc", "temperature", et0, day)
    _write_monthly_nc(root / "Temperature_min" / "temperature_min_2015_04.nc", "temperature_min", et0, day)
    _write_monthly_nc(root / "Temperature_max" / "temperature_max_2015_04.nc", "temperature_max", et0, day)

    model = DummyModel()
    reader = MonthlyNetCDFForcingReader.from_dao_layout(
        root,
        start_date=day,
        end_date=day,
    )

    season = {
        "phenoe1": np.full((2, 2), 9, dtype=np.float32),
        "phenom1": np.full((2, 2), 10, dtype=np.float32),
        "phenos1": np.full((2, 2), 11, dtype=np.float32),
        "phenosen1": np.full((2, 2), 12, dtype=np.float32),
    }

    runner = MonthlyIWRRunner(
        model=model,
        forcing_reader=reader,
        crop_calendar=_build_calendar(),
        crop_name="wheat",
        output_dir=output,
        write_monthly_outputs=True,
        use_phenology_kc=True,
        phenology_layers=season,
    )

    result = runner.run()
    assert len(result.written_files) == 1

    ds = xr.open_dataset(result.written_files[0])
    assert "kc" in ds
    assert "phenology_stage" in ds
    assert int(ds["phenology_stage"].values[0, 0, 0]) == int(PHENOLOGY_STAGE_MAXIMUM)
    assert np.isclose(float(ds["kc"].values[0, 0, 0]), 1.0)
    ds.close()


def test_monthly_runner_skip_inactive_sets_zero_etc(tmp_path):
    root = tmp_path / "forcing"
    output = tmp_path / "out"
    for folder in ["ET_HS", "P", "Temperature", "Temperature_min", "Temperature_max"]:
        (root / folder).mkdir(parents=True, exist_ok=True)

    day = date(2015, 4, 10)
    et0 = np.stack([np.full((2, 2), 2.0, dtype=np.float32)])
    p = np.stack([np.zeros((2, 2), dtype=np.float32)])

    _write_monthly_nc(root / "ET_HS" / "potential_evapotranspiration_2015_04.nc", "PET", et0, day)
    _write_monthly_nc(root / "P" / "precipitation_2015_04.nc", "precipitation", p, day)
    _write_monthly_nc(root / "Temperature" / "temperature_2015_04.nc", "temperature", et0, day)
    _write_monthly_nc(root / "Temperature_min" / "temperature_min_2015_04.nc", "temperature_min", et0, day)
    _write_monthly_nc(root / "Temperature_max" / "temperature_max_2015_04.nc", "temperature_max", et0, day)

    model = DummyModel()
    reader = MonthlyNetCDFForcingReader.from_dao_layout(
        root,
        start_date=day,
        end_date=day,
    )

    # Inactive at dekad 10 (season starts at 20)
    season = {
        "phenoe1": np.full((2, 2), 20, dtype=np.float32),
        "phenom1": np.full((2, 2), 21, dtype=np.float32),
        "phenos1": np.full((2, 2), 22, dtype=np.float32),
        "phenosen1": np.full((2, 2), 23, dtype=np.float32),
    }

    runner = MonthlyIWRRunner(
        model=model,
        forcing_reader=reader,
        crop_calendar=_build_calendar(),
        crop_name="wheat",
        output_dir=output,
        write_monthly_outputs=True,
        use_phenology_kc=True,
        skip_iwr_when_inactive=True,
        phenology_layers=season,
    )

    result = runner.run()
    ds = xr.open_dataset(result.written_files[0])

    # etc=0 -> dummy model keeps soil unchanged from initial 10 mm
    assert np.isclose(float(ds["soil_moisture_next_mm"].values[0, 0, 0]), 10.0)
    assert int(ds["phenology_stage"].values[0, 0, 0]) == int(PHENOLOGY_STAGE_INACTIVE)
    ds.close()
