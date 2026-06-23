from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from iwr_processing.preprocessing_functions import (
    aggregate_fractional_layers,
    build_crop_calendar_template_from_raster,
)


def _write_raster(path: Path, data: np.ndarray, nodata: float | int) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=str(data.dtype),
        crs="EPSG:4326",
        transform=from_origin(0, data.shape[0], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_aggregate_fractional_layers_preserves_nodata_and_normalizes_descriptions(tmp_path):
    source = tmp_path / "landcover.tif"
    grid = tmp_path / "grid.tif"
    output = tmp_path / "fractions.tif"

    _write_raster(
        source,
        np.array([[1, 1], [2, -9999]], dtype=np.int16),
        nodata=-9999,
    )
    _write_raster(
        grid,
        np.array([[1, 1], [1, 1]], dtype=np.int16),
        nodata=-9999,
    )

    result_path = aggregate_fractional_layers(
        grid=str(grid),
        class_values=[1, 2],
        classes=["Crop A", "Crop-B"],
        land_cover=str(source),
        output=str(output),
        output_nodata=-9999.0,
    )

    with rasterio.open(result_path) as src:
        band_1 = src.read(1)
        band_2 = src.read(2)
        assert src.descriptions == ("frac_crop_a", "frac_crop_b")
        assert np.isclose(band_1[1, 1], -9999.0)
        assert np.isclose(band_2[1, 1], -9999.0)
        assert np.isclose(band_1[0, 0], 100.0)
        assert np.isclose(band_2[0, 0], 0.0)


def test_build_crop_calendar_template_from_fractional_raster(tmp_path):
    raster_path = tmp_path / "land_cover_fractional.tif"
    output_csv = tmp_path / "crop_calendar_template.csv"

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.array([[20.0, 0.0], [0.0, -9999.0]], dtype=np.float32), 1)
        dst.write(np.array([[0.0, 50.0], [0.0, -9999.0]], dtype=np.float32), 2)
        dst.set_band_description(1, "frac_winter_wheat")
        dst.set_band_description(2, "frac_maize_field_grain_field_corn")

    result_path = build_crop_calendar_template_from_raster(
        land_cover_raster=str(raster_path),
        output=str(output_csv),
    )

    import pandas as pd

    df = pd.read_csv(result_path)
    assert "year" not in df.columns
    assert df["crop_id"].tolist() == [
        "maize_field_grain_field_corn",
        "winter_wheat",
    ]
    assert np.isnan(df.loc[df["crop_id"] == "winter_wheat", "planting_doy"]).all()
    assert np.isnan(df.loc[df["crop_id"] == "winter_wheat", "harvest_doy"]).all()
    assert np.isnan(df.loc[df["crop_id"] == "winter_wheat", "kc_ini"]).all()
    assert np.isnan(df.loc[df["crop_id"] == "winter_wheat", "kc_mid"]).all()
    assert np.isnan(df.loc[df["crop_id"] == "winter_wheat", "kc_end"]).all()