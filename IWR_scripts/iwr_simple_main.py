import argparse
import json
import os
from pathlib import Path
from datetime import datetime


# Prevent cross-environment PROJ database conflicts (e.g. active conda env + .venv python).
for _proj_var in ("PROJ_LIB", "PROJ_DATA"):
    _proj_path = os.environ.get(_proj_var)
    if _proj_path and "miniconda3/envs/" in _proj_path:
        os.environ.pop(_proj_var, None)

import rasterio

from soil_functions import create_soil_parameter_rasters, check_cached_soil_parameters_metadata
from crop_functions import check_crop_raster_and_csv
from phenology_functions import load_phenology_layers
from iwr_model import (
    normalize_iwr_configuration,
    normalize_drainage_scheme,
    resolve_offseason_water_balance_kc,
    run_iwr_model,
)
from checks import run_input_checks


_MISSING = object()


def _nested_get(config, path, default=_MISSING):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _cfg_get(config, nested_path, fallback_keys=(), default=_MISSING):
    value = _nested_get(config, nested_path, default=_MISSING)
    if value is not _MISSING:
        return value

    for key in fallback_keys:
        if key in config:
            return config[key]

    if default is not _MISSING:
        return default

    tried = [nested_path, *fallback_keys]
    raise KeyError(f"Missing required config key(s): {tried}")


def read_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def read_raster(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def parse_args():
    default_config_path = Path(__file__).resolve().parent / "config" / "config_eraL.json"

    parser = argparse.ArgumentParser(
        description="Run the simple IWR workflow from a JSON configuration file."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(default_config_path),
        help=(
            "Path to the JSON config file. "
            f"Defaults to {default_config_path}"
        ),
    )
    return parser.parse_args()


def main(config_path=None):
    if config_path is None:
        args = parse_args()
        config_path = args.config_path

    config_path = Path(config_path).expanduser().resolve()
    config = read_config(config_path)

    start_date = datetime.strptime(
        _cfg_get(config, "time.start_date", fallback_keys=("start_date",)),
        "%Y-%m-%d",
    )
    end_date = datetime.strptime(
        _cfg_get(config, "time.end_date", fallback_keys=("end_date",)),
        "%Y-%m-%d",
    )

    _spinup_start_date_str = _cfg_get(
        config,
        "time.spinup_start_date",
        fallback_keys=("spinup_start_date",),
        default=None,
    )
    if _spinup_start_date_str not in (None, ""):
        spinup_start_date = datetime.strptime(_spinup_start_date_str, "%Y-%m-%d")
        if spinup_start_date > start_date:
            raise ValueError(
                f"spinup_start_date ({_spinup_start_date_str}) must be <= "
                f"start_date ({start_date.strftime('%Y-%m-%d')})."
            )
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date.strftime('%Y-%m-%d')}) must be <= "
                f"end_date ({end_date.strftime('%Y-%m-%d')})."
            )
    else:
        spinup_start_date = None

    iwr_mode = _cfg_get(
        config,
        "options.iwr_mode",
        fallback_keys=("iwr_mode",),
        default="watneeds_blue_et",
    )
    iwr_domain = _cfg_get(
        config,
        "options.iwr_domain",
        fallback_keys=("iwr_domain",),
        default=None,
    )
    theoretical_iwr_target = _cfg_get(
        config,
        "options.theoretical_iwr_target",
        fallback_keys=("theoretical_iwr_target",),
        default="stress_threshold",
    )
    initial_soil_moisture_fraction = _cfg_get(
        config,
        "options.initial_soil_moisture_fraction",
        fallback_keys=("initial_soil_moisture_fraction",),
        default=0.5,
    )
    offseason_water_balance_kc = _cfg_get(
        config,
        "options.offseason_water_balance_kc",
        fallback_keys=("offseason_water_balance_kc",),
        default=None,
    )

    iwr_mode, iwr_domain, theoretical_iwr_target = normalize_iwr_configuration(
        iwr_mode=iwr_mode,
        iwr_domain=iwr_domain,
        theoretical_iwr_target=theoretical_iwr_target,
    )

    resolved_offseason_water_balance_kc = resolve_offseason_water_balance_kc(
        iwr_mode=iwr_mode,
        offseason_water_balance_kc=offseason_water_balance_kc,
    )

    drainage_scheme = _cfg_get(
        config,
        "options.drainage_scheme",
        fallback_keys=("drainage_scheme",),
        default="auto",
    )

    resolved_drainage_scheme = normalize_drainage_scheme(
        iwr_mode=iwr_mode,
        drainage_scheme=drainage_scheme,
    )

    print(
        "Selected IWR configuration:",
        f"mode={iwr_mode},",
        f"domain={iwr_domain},",
        f"target={theoretical_iwr_target},",
        f"initial_soil_moisture_fraction={float(initial_soil_moisture_fraction):.3f},",
        f"offseason_water_balance_kc={resolved_offseason_water_balance_kc:.3f},",
        f"drainage_scheme={resolved_drainage_scheme}",
    )

    irrigated_areas_value = _cfg_get(
        config,
        "datasets.static.irrigated_areas_path",
        fallback_keys=("irrigated_areas_path",),
        default=None,
    )
    irrigated_areas_path = (
        Path(irrigated_areas_value)
        if irrigated_areas_value not in (None, "")
        else None
    )
    valid_mask_path = Path(
        _cfg_get(config, "datasets.static.valid_mask_path", fallback_keys=("valid_mask_path",))
    )
    soil_texture_path = Path(
        _cfg_get(
            config,
            "datasets.static.soil_texture_path",
            fallback_keys=("soil_texture_path",),
        )
    )

    phenology_paths = {
        name: Path(path)
        for name, path in _cfg_get(
            config,
            "datasets.static.phenology_paths",
            fallback_keys=("phenology_paths",),
        ).items()
    }
    phenology = load_phenology_layers(phenology_paths)

    precipitation_geotiff_folder = Path(
        _cfg_get(
            config,
            "datasets.forcing.precipitation_geotiff_folder",
            fallback_keys=("precipitation_geotiff_folder", "precipitation_folder"),
        )
    )
    et0_geotiff_folder = Path(
        _cfg_get(
            config,
            "datasets.forcing.et0_geotiff_folder",
            fallback_keys=("et0_geotiff_folder", "et0_folder"),
        )
    )

    soil_output_folder = Path(
        _cfg_get(config, "outputs.soil_output_folder", fallback_keys=("soil_output_folder",))
    )

    force_regenerate_soil_parameters = _cfg_get(
        config,
        "options.force_regenerate_soil_parameters",
        fallback_keys=("force_regenerate_soil_parameters",),
        default=False,
    )

    soil_outputs = {
        "field_capacity": soil_output_folder / "field_capacity.tif",
        "wilting_point": soil_output_folder / "wilting_point.tif",
        "total_available_water": soil_output_folder / "total_available_water.tif",
        "fmax": soil_output_folder / "fmax.tif",
    }

    files_present = all(path.exists() for path in soil_outputs.values())

    if force_regenerate_soil_parameters or not files_present:
        if force_regenerate_soil_parameters:
            print(
                "Soil parameter rasters: regenerating (force_regenerate_soil_parameters=true)."
            )
        else:
            missing = [p.name for p in soil_outputs.values() if not p.exists()]
            print(
                f"Soil parameter rasters: regenerating (missing file(s): {missing})."
            )
        soil_outputs = create_soil_parameter_rasters(
            soil_texture_path=soil_texture_path,
            output_folder=soil_output_folder,
        )
        print("Soil parameter rasters: regeneration complete.")
    else:
        print(
            "Soil parameter rasters: reusing cached files in "
            f"{soil_output_folder}."
        )
        check_cached_soil_parameters_metadata(soil_output_folder)

    total_available_water, output_profile = read_raster(
        soil_outputs["total_available_water"]
    )

    fmax, _ = read_raster(
        soil_outputs["fmax"]
    )

    output_base = Path(_cfg_get(config, "outputs.output_base", fallback_keys=("output_base",)))
    run_name = _cfg_get(config, "outputs.run_name", fallback_keys=("run_name",))
    iwr_output_folder = output_base / run_name
    debug_output_folder = str(output_base / f"{run_name}_debug")

    if irrigated_areas_path is not None:
        irrigation_mask, irrigation_profile = read_raster(irrigated_areas_path)
    else:
        irrigation_mask = None
        irrigation_profile = None

    valid_area_mask, valid_area_profile = read_raster(valid_mask_path)

    crop_fraction_path = Path(
        _cfg_get(
            config,
            "datasets.static.crop.crop_fraction_path",
            fallback_keys=("crop_fraction_path",),
        )
    )
    crop_parameters_csv = Path(
        _cfg_get(
            config,
            "datasets.static.crop.crop_parameters_csv",
            fallback_keys=("crop_parameters_csv",),
        )
    )

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
        valid_area_mask=valid_area_mask,
        valid_area_profile=valid_area_profile,
        crop_fraction_data=crop_fraction_data,
        crop_profile=crop_profile,
        phenology=phenology,
    )

    write_daily_etx = _cfg_get(
        config,
        "options.write_daily_etx",
        fallback_keys=("write_daily_etx",),
        default=False,
    )

    write_daily_eta_stress = _cfg_get(
        config,
        "options.write_daily_eta_stress",
        fallback_keys=("write_daily_eta_stress",),
        default=False,
    )

    write_static_support_layers = _cfg_get(
        config,
        "options.write_static_support_layers",
        fallback_keys=("write_static_support_layers",),
        default=True,
    )

    inactive_kc = _cfg_get(
        config,
        "options.inactive_kc",
        fallback_keys=("inactive_kc",),
        default=0.0,
    )

    print("Write daily ETx:", write_daily_etx)
    print("Write daily stress-limited ETa:", write_daily_eta_stress)
    print("Write static support layers:", write_static_support_layers)
    print("Inactive Kc (outside active phenology):", inactive_kc)

    final_soil_moisture, cumulative_irrigation = run_iwr_model(
        start_date=start_date,
        end_date=end_date,
        total_available_water=total_available_water,
        fmax=fmax,
        irrigation_mask=irrigation_mask,
        valid_area_mask=valid_area_mask,
        crop_fraction_data=crop_fraction_data,
        crop_df=crop_df,
        phenology=phenology,
        precipitation_geotiff_folder=precipitation_geotiff_folder,
        et0_geotiff_folder=et0_geotiff_folder,
        output_folder=iwr_output_folder,
        output_profile=output_profile,
        strict_checks=_cfg_get(config, "options.strict_checks", fallback_keys=("strict_checks",), default=True),
        write_debug_csv=_cfg_get(config, "options.write_debug_csv", fallback_keys=("write_debug_csv",), default=True),
        write_cumulative_iwr=_cfg_get(
            config,
            "options.write_cumulative_iwr",
            fallback_keys=("write_cumulative_iwr",),
            default=True,
        ),
        write_green_blue_outputs=False,
        write_daily_green_blue_outputs=False,
        write_daily_etx=write_daily_etx,
        write_daily_eta_stress=write_daily_eta_stress,
        write_static_support_layers=write_static_support_layers,
        write_active_pixel_masks=_cfg_get(
            config,
            "options.write_active_pixel_masks",
            fallback_keys=("write_active_pixel_masks",),
            default=False,
        ),
        debug_mode=_cfg_get(config, "options.debug_mode", fallback_keys=("debug_mode",), default=False),
        debug_output_folder=debug_output_folder,
        max_precipitation_mm_day=_cfg_get(
            config,
            "options.max_precipitation_mm_day",
            fallback_keys=("max_precipitation_mm_day",),
            default=300,
        ),
        max_et0_mm_day=_cfg_get(
            config,
            "options.max_et0_mm_day",
            fallback_keys=("max_et0_mm_day",),
            default=20,
        ),
        max_iwr_mm_day=_cfg_get(
            config,
            "options.max_iwr_mm_day",
            fallback_keys=("max_iwr_mm_day",),
            default=100,
        ),
        min_valid_forcing_fraction=_cfg_get(
            config,
            "options.min_valid_forcing_fraction",
            fallback_keys=("min_valid_forcing_fraction",),
            default=0.01,
        ),
        debug_csv_frequency_days=_cfg_get(
            config,
            "options.debug_csv_frequency_days",
            fallback_keys=("debug_csv_frequency_days",),
            default=1,
        ),
        iwr_mode=iwr_mode,
        iwr_domain=iwr_domain,
        theoretical_iwr_target=theoretical_iwr_target,
        initial_soil_moisture_fraction=initial_soil_moisture_fraction,
        inactive_kc=inactive_kc,
        offseason_water_balance_kc=offseason_water_balance_kc,
        drainage_scheme=drainage_scheme,
        spinup_start_date=spinup_start_date,
    )

    print("Configuration loaded")
    if spinup_start_date is not None:
        print("Spin-up start date:", spinup_start_date)
    print("Start date:", start_date)
    print("End date:", end_date)
    print("IWR mode:", iwr_mode)
    print("IWR domain:", iwr_domain)
    print("Theoretical IWR target:", theoretical_iwr_target)
    print("Initial soil moisture fraction:", initial_soil_moisture_fraction)
    print("Inactive Kc:", inactive_kc)
    print("Off-season water-balance Kc:", resolved_offseason_water_balance_kc)
    print("Drainage scheme:", resolved_drainage_scheme)
    print("Irrigated areas:", irrigated_areas_path)
    print("Valid mask:", valid_mask_path)
    print("Soil texture:", soil_texture_path)
    print("Phenology:", phenology_paths)
    print("Phenology layers loaded:", list(phenology.keys()))
    print("Precipitation folder:", precipitation_geotiff_folder)
    print("ET0 folder:", et0_geotiff_folder)

    print("Soil parameter rasters created:")
    for name, path in soil_outputs.items():
        print(name, ":", path)

    print("Crop parameters loaded")
    if _cfg_get(config, "options.print_crop_parameters", fallback_keys=("print_crop_parameters",), default=False):
        print(crop_df)
    print("Crop fraction raster shape:", crop_fraction_data.shape)
    print("Crop raster bands:", crop_band_descriptions)

    print("IWR model skeleton completed")
    print("Final soil moisture shape:", final_soil_moisture.shape)
    print("Cumulative irrigation shape:", cumulative_irrigation.shape)


if __name__ == "__main__":
    main()