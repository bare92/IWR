import json
from pathlib import Path
from datetime import datetime

import rasterio

from soil_functions import create_soil_parameter_rasters
from crop_functions import check_crop_raster_and_csv
from phenology_functions import load_phenology_layers
from iwr_model import run_iwr_model
from checks import run_input_checks


def read_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def read_raster(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def main():
    config = read_config("config/config.json")

    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(config["end_date"], "%Y-%m-%d")

    irrigated_areas_path = Path(config["irrigated_areas_path"])
    land_cover_fractions_path = Path(config["land_cover_fractions_path"])
    soil_texture_path = Path(config["soil_texture_path"])

    phenology_paths = {
        name: Path(path)
        for name, path in config["phenology_paths"].items()
    }
    phenology = load_phenology_layers(phenology_paths)

    precipitation_geotiff_folder = Path(config["precipitation_geotiff_folder"])
    et0_geotiff_folder = Path(config["et0_geotiff_folder"])

    soil_output_folder = Path(config["soil_output_folder"])

    soil_outputs = create_soil_parameter_rasters(
        soil_texture_path=soil_texture_path,
        output_folder=soil_output_folder,
    )

    total_available_water, output_profile = read_raster(
        soil_outputs["total_available_water"]
    )

    fmax, _ = read_raster(
        soil_outputs["fmax"]
    )

    iwr_output_folder = Path(config["iwr_output_folder"])

    irrigation_mask, irrigation_profile = read_raster(irrigated_areas_path)

    crop_fraction_path = Path(config["crop_fraction_path"])
    crop_parameters_csv = Path(config["crop_parameters_csv"])

    crop_df, crop_fraction_data, crop_profile, crop_band_descriptions = check_crop_raster_and_csv(
        crop_fraction_path=crop_fraction_path,
        crop_parameters_csv=crop_parameters_csv,
    )

    run_input_checks(
        reference_shape=total_available_water.shape,
        reference_profile=output_profile,
        fmax=fmax,
        irrigation_mask=irrigation_mask,
        irrigation_profile=irrigation_profile,
        crop_fraction_data=crop_fraction_data,
        crop_profile=crop_profile,
        phenology=phenology,
    )

    final_soil_moisture, cumulative_irrigation = run_iwr_model(
        start_date=start_date,
        end_date=end_date,
        total_available_water=total_available_water,
        fmax=fmax,
        irrigation_mask=irrigation_mask,
        crop_fraction_data=crop_fraction_data,
        crop_df=crop_df,
        phenology=phenology,
        precipitation_geotiff_folder=precipitation_geotiff_folder,
        et0_geotiff_folder=et0_geotiff_folder,
        output_folder=iwr_output_folder,
        output_profile=output_profile,
    )

    print("Configuration loaded")
    print("Start date:", start_date)
    print("End date:", end_date)
    print("Irrigated areas:", irrigated_areas_path)
    print("Land cover fractions:", land_cover_fractions_path)
    print("Soil texture:", soil_texture_path)
    print("Phenology:", phenology_paths)
    print("Phenology layers loaded:", list(phenology.keys()))
    print("Precipitation folder:", precipitation_geotiff_folder)
    print("ET0 folder:", et0_geotiff_folder)

    print("Soil parameter rasters created:")
    for name, path in soil_outputs.items():
        print(name, ":", path)

    print("Crop parameters loaded")
    print(crop_df)
    print("Crop fraction raster shape:", crop_fraction_data.shape)
    print("Crop raster bands:", crop_band_descriptions)

    print("IWR model skeleton completed")
    print("Final soil moisture shape:", final_soil_moisture.shape)
    print("Cumulative irrigation shape:", cumulative_irrigation.shape)


if __name__ == "__main__":
    main()