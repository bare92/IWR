import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IWR_scripts"))

from iwr_simple_main import main  # noqa: E402

NODATA = -9999.0


def write_tif(path, data, profile):
    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data.astype(np.float32), 1)
        else:
            dst.write(data.astype(np.float32))


def build_profile(height, width, count=1):
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": "float32",
        "nodata": NODATA,
        "crs": "EPSG:3035",
        "transform": Affine(1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0),
        "compress": "lzw",
    }


def test_main_with_null_irrigation_path():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        static_dir = td / "static"
        forcing_p = td / "forcing_p"
        forcing_pet = td / "forcing_pet"
        output_base = td / "outputs"
        static_dir.mkdir()
        forcing_p.mkdir()
        forcing_pet.mkdir()
        output_base.mkdir()

        shape = (1, 1)
        profile1 = build_profile(*shape, count=1)

        valid_mask = np.array([[1.0]], dtype=np.float32)
        soil_texture = np.array([[9.0]], dtype=np.float32)

        write_tif(static_dir / "valid_mask.tif", valid_mask, profile1)
        write_tif(static_dir / "soil_texture.tif", soil_texture, profile1)

        phenology_names = [
            "phenoe1",
            "phenoe2",
            "phenom1",
            "phenom2",
            "phenonseasons",
            "phenos1",
            "phenos2",
            "phenosen1",
            "phenosen2",
        ]
        for name in phenology_names:
            if name == "phenonseasons":
                arr = np.array([[1.0]], dtype=np.float32)
            elif name in ("phenos1",):
                arr = np.array([[1.0]], dtype=np.float32)
            elif name in ("phenom1",):
                arr = np.array([[2.0]], dtype=np.float32)
            elif name in ("phenosen1",):
                arr = np.array([[3.0]], dtype=np.float32)
            elif name in ("phenoe1",):
                arr = np.array([[36.0]], dtype=np.float32)
            else:
                arr = np.array([[0.0]], dtype=np.float32)
            write_tif(static_dir / f"{name}.tif", arr, profile1)

        precip = np.array([[0.0]], dtype=np.float32)
        et0 = np.array([[5.0]], dtype=np.float32)
        write_tif(forcing_p / "P_20210101.tif", precip, profile1)
        write_tif(forcing_pet / "PET_20210101.tif", et0, profile1)

        crop_fraction_profile = build_profile(*shape, count=1)
        crop_fraction_data = np.array([[[1.0]]], dtype=np.float32)
        write_tif(static_dir / "crop_fraction.tif", crop_fraction_data, crop_fraction_profile)

        crop_csv = static_dir / "crop_params.csv"
        crop_csv.write_text(
            "crop_name,root_depth_max_m,Kc_ini,Kc_mid,Kc_end,p\n"
            "crop_a,1.0,1.0,1.0,1.0,0.5\n",
            encoding="utf-8",
        )

        config = {
            "start_date": "2021-01-01",
            "end_date": "2021-01-01",
            "iwr_mode": "theoretical_net_irrigation",
            "iwr_domain": "all_cropped",
            "theoretical_iwr_target": "stress_threshold",
            "initial_soil_moisture_fraction": 0.5,
            "irrigated_areas_path": None,
            "valid_mask_path": str(static_dir / "valid_mask.tif"),
            "soil_texture_path": str(static_dir / "soil_texture.tif"),
            "phenology_paths": {name: str(static_dir / f"{name}.tif") for name in phenology_names},
            "precipitation_geotiff_folder": str(forcing_p),
            "et0_geotiff_folder": str(forcing_pet),
            "soil_output_folder": str(td / "soil_outputs"),
            "output_base": str(output_base),
            "run_name": "IWR_theoretical_test",
            "crop_fraction_path": str(static_dir / "crop_fraction.tif"),
            "crop_parameters_csv": str(crop_csv),
            "strict_checks": False,
            "write_debug_csv": False,
            "write_cumulative_iwr": True,
            "debug_mode": False
        }

        cfg_path = td / "config.json"
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        main(str(cfg_path))

        daily_output = output_base / "IWR_theoretical_test" / "IWR" / "iwr_20210101.tif"
        assert daily_output.exists(), "Expected daily IWR output GeoTIFF to be written"

        with rasterio.open(daily_output) as src:
            tags = src.tags()

        assert tags.get("iwr_mode") == "theoretical_net_irrigation"
        assert tags.get("iwr_domain") == "all_cropped"
        assert tags.get("variable") == "daily_theoretical_net_irrigation_requirement"


def main_script():
    test_main_with_null_irrigation_path()
    print("smoke_test_main_rainfed_config.py: PASS")


if __name__ == "__main__":
    main_script()
