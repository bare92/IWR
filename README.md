# IWR Processing

Irrigation Water Requirement processing project for gridded geospatial workflows.

This project uses `pyproject.toml` for dependency management.

## IWR Modes

The simple workflow supports two calculation modes:

- `watneeds_blue_et` (legacy): daily blue ET gap on irrigated pixels only.
- `theoretical_net_irrigation`: daily net irrigation requirement from a separate reference soil-water state.

## Example Theoretical Rainfed Config

A template config is provided at `IWR_scripts/config/config_eraL_theoretical_rainfed.json`.

Important:
- The placeholder paths `/path/to/rainfed_crop_fraction_raster.tif` and `/path/to/rainfed_crop_parameters.csv` must be replaced with real server paths before running.
- If `crop_fraction_path` contains total cropland (not rainfed-only), keep `irrigated_areas_path` and set `iwr_domain` to `non_irrigated`.
