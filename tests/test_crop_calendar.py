from datetime import date

import numpy as np
import pandas as pd
import pytest

from iwr_processing.crop_calendar import (
    CropCalendar,
    CropGrowthSchedule,
    GrowthStage,
    TimeSeriesDriver,
    compute_kc_daily,
    load_crop_calendars_from_csv,
)


def build_calendar() -> CropCalendar:
    return CropCalendar(
        crop_id="wheat",
        year=2020,
        planting_doy=100,
        harvest_doy=109,
        growth_schedule=CropGrowthSchedule(
            crop_id="wheat",
            kc_ini=0.4,
            kc_mid=1.2,
            kc_end=0.6,
            growth_stage_days={
                GrowthStage.INITIAL: 2,
                GrowthStage.DEVELOPMENT: 3,
                GrowthStage.MID_SEASON: 3,
                GrowthStage.LATE_SEASON: 2,
            },
        ),
    )


def test_compute_kc_daily_interpolates_stages():
    calendar = build_calendar()
    assert compute_kc_daily(calendar, 99, 2020) == pytest.approx(0.5)
    assert compute_kc_daily(calendar, 100, 2020) == pytest.approx(0.4)
    assert compute_kc_daily(calendar, 101, 2020) == pytest.approx(0.4)
    assert compute_kc_daily(calendar, 103, 2020) == pytest.approx(0.8)
    assert compute_kc_daily(calendar, 104, 2020) == pytest.approx(1.2)
    assert compute_kc_daily(calendar, 108, 2020) == pytest.approx(1.2)
    assert compute_kc_daily(calendar, 109, 2020) == pytest.approx(0.6)


def test_load_crop_calendars_from_csv_validates_lengths(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "crop_id": "wheat",
                "year": 2020,
                "planting_doy": 100,
                "harvest_doy": 104,
                "kc_ini": 0.4,
                "kc_mid": 1.2,
                "kc_end": 0.6,
                "growth_stage_days_initial": 2,
                "growth_stage_days_development": 2,
                "growth_stage_days_mid_season": 2,
                "growth_stage_days_late_season": 2,
            }
        ]
    )
    csv_path = tmp_path / "calendar.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_crop_calendars_from_csv(str(csv_path))


def test_load_crop_calendars_from_yearless_csv_expands_years(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "crop_id": "wheat",
                "planting_doy": 100,
                "harvest_doy": 109,
                "kc_ini": 0.4,
                "kc_mid": 1.2,
                "kc_end": 0.6,
                "growth_stage_days_initial": 2,
                "growth_stage_days_development": 3,
                "growth_stage_days_mid_season": 3,
                "growth_stage_days_late_season": 2,
            }
        ]
    )
    csv_path = tmp_path / "calendar_yearless.csv"
    frame.to_csv(csv_path, index=False)

    calendars = load_crop_calendars_from_csv(str(csv_path), years=[2020, 2021])
    assert ("wheat", 2020) in calendars
    assert ("wheat", 2021) in calendars


class DummyCropFractions:
    crop_type = ["wheat"]

    def get_crop_fraction(self, crop: str) -> np.ndarray:
        assert crop == "wheat"
        return np.array([[50.0]], dtype=np.float32)


class DummyModel:
    def __init__(self):
        self.crop_fractions = DummyCropFractions()
        self.peff_reduction_pct = 5.0
        self.spinup_calls = 0

    def get_crop_initial_soil_water_mm(self, crop: str) -> np.ndarray:
        assert crop == "wheat"
        return np.array([[10.0]], dtype=np.float32)

    def watneeds_spinup(self, **kwargs):
        self.spinup_calls += 1
        assert "precipitation_mm_day_series" in kwargs
        return {"soil_moisture_spinup_end_mm": np.array([[12.0]], dtype=np.float32)}

    def green_water_step(self, **kwargs):
        assert "precipitation_mm" in kwargs
        assert "etc_mm" in kwargs
        return {
            "s_next_mm": np.array([[11.0]], dtype=np.float32),
            "green_et_mm": np.array([[2.0]], dtype=np.float32),
            "blue_water_mm": np.array([[1.0]], dtype=np.float32),
        }


def test_time_series_driver_runs_and_weights_outputs():
    calendar = build_calendar()
    model = DummyModel()
    start_date = date(2020, 4, 9)
    end_date = date(2020, 4, 11)
    precipitation = {
        start_date: 1.0,
        date(2020, 4, 10): 0.0,
        end_date: 2.0,
    }
    et0 = {
        start_date: 4.0,
        date(2020, 4, 10): 4.0,
        end_date: 4.0,
    }

    driver = TimeSeriesDriver(
        iwr_model=model,
        crop_calendars={("wheat", 2020): calendar},
        start_date=start_date,
        end_date=end_date,
        precipitation_by_date=precipitation,
        et0_by_date=et0,
        spinup_years=1,
    )

    outputs = driver.run()
    assert model.spinup_calls == 1
    assert len(outputs) == 3
    assert outputs[0]["green_water_mm"][0, 0] == pytest.approx(1.0)
    assert outputs[0]["blue_water_mm"][0, 0] == pytest.approx(0.5)
    monthly = driver.get_monthly_aggregates()
    assert monthly[("wheat", 2020, 4)]["green_water_mm"][0, 0] == pytest.approx(3.0)