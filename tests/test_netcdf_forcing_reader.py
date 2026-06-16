from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from iwr_processing.netcdf_forcing_reader import MonthlyNetCDFForcingReader


def _write_monthly_nc(path, var_name, values, start_date):
    time = pd.date_range(start=start_date, periods=values.shape[0], freq="D")
    ds = xr.Dataset(
        {var_name: (("time", "y", "x"), values.astype(np.float32))},
        coords={"time": time, "y": np.arange(values.shape[1]), "x": np.arange(values.shape[2])},
    )
    ds.to_netcdf(path)


def test_dao_monthly_netcdf_reader_returns_daily_arrays(tmp_path):
    root = tmp_path
    et0_dir = root / "ET_HS"
    p_dir = root / "P"
    t_dir = root / "Temperature"
    tmin_dir = root / "Temperature_min"
    tmax_dir = root / "Temperature_max"
    et0_dir.mkdir()
    p_dir.mkdir()
    t_dir.mkdir()
    tmin_dir.mkdir()
    tmax_dir.mkdir()

    et0_values = np.stack([
        np.full((2, 2), 1.0, dtype=np.float32),
        np.full((2, 2), 2.0, dtype=np.float32),
        np.full((2, 2), 3.0, dtype=np.float32),
    ])
    p_values = np.stack([
        np.full((2, 2), 10.0, dtype=np.float32),
        np.full((2, 2), 20.0, dtype=np.float32),
        np.full((2, 2), 30.0, dtype=np.float32),
    ])

    _write_monthly_nc(et0_dir / "potential_evapotranspiration_2015_01.nc", "PET", et0_values, date(2015, 1, 1))
    _write_monthly_nc(p_dir / "precipitation_2015_01.nc", "precipitation", p_values, date(2015, 1, 1))
    _write_monthly_nc(t_dir / "temperature_2015_01.nc", "temperature", et0_values + 10.0, date(2015, 1, 1))
    _write_monthly_nc(tmin_dir / "temperature_min_2015_01.nc", "temperature_min", et0_values + 5.0, date(2015, 1, 1))
    _write_monthly_nc(tmax_dir / "temperature_max_2015_01.nc", "temperature_max", et0_values + 15.0, date(2015, 1, 1))

    reader = MonthlyNetCDFForcingReader.from_dao_layout(
        root_dir=root,
        start_date=date(2015, 1, 1),
        end_date=date(2015, 1, 3),
    )

    day1 = reader.get_day(date(2015, 1, 1))
    day2 = reader.get_day(date(2015, 1, 2))

    assert np.allclose(day1["et0"], 1.0)
    assert np.allclose(day2["et0"], 2.0)
    assert np.allclose(day1["precipitation"], 10.0)
    assert np.allclose(day2["precipitation"], 20.0)
    assert np.allclose(day1["temperature"], 11.0)
    assert np.allclose(day1["temperature_min"], 6.0)
    assert np.allclose(day1["temperature_max"], 16.0)

    daily = list(reader.iter_daily())
    assert len(daily) == 3
    assert daily[2][0] == date(2015, 1, 3)
    assert np.allclose(daily[2][1]["et0"], 3.0)