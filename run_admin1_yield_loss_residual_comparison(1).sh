#!/usr/bin/env bash

# Launch the Admin-1 comparison between FAO33 yield-loss maps and
# residual-derived common-crop yields.

set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHON_SCRIPT="${PYTHON_SCRIPT:-/home/fremen/workspaces/RB/IWR/scripts_additional/compare_admin1_yield_loss_with_residuals.py}"
RESIDUAL_CSV="${RESIDUAL_CSV:-/home/fremen/data/projects/Burkina/Burkina_Faso_admin1/residuals_data.csv}"
CROP_CONFIG_CSV="${CROP_CONFIG_CSV:-/home/fremen/data/projects/Burkina/00_Data_iwr/static/CROPG_fractional.csv}"
ADMIN1_VECTOR="${ADMIN1_VECTOR:-/home/fremen/data/basedata/BURKINA_FASO/shp/bfaadmbndaadm11msalbitos.zip}"
YIELD_LOSS_DIR="${YIELD_LOSS_DIR:-/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output_260903/IWR_theoretical_rainfed/Seasonal_Yield_Loss_FAO33/Yield_loss_fraction}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/fremen/data/projects/Burkina/00_Data_iwr/iwr_output_260903/IWR_theoretical_rainfed/Seasonal_Yield_Loss_FAO33/Admin1_residual_comparison}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
else
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
        fail "Python executable not found: ${PYTHON_BIN}"
fi

[[ -f "${PYTHON_SCRIPT}" ]] || fail "Python script not found: ${PYTHON_SCRIPT}"
[[ -f "${RESIDUAL_CSV}" ]] || fail "Residual CSV not found: ${RESIDUAL_CSV}"
[[ -f "${CROP_CONFIG_CSV}" ]] || fail "Crop configuration CSV not found: ${CROP_CONFIG_CSV}"
[[ -f "${ADMIN1_VECTOR}" ]] || fail "Admin-1 ZIP not found: ${ADMIN1_VECTOR}"
[[ -d "${YIELD_LOSS_DIR}" ]] || fail "Yield-loss directory not found: ${YIELD_LOSS_DIR}"

printf 'Running Admin-1 yield-loss comparison\n'
printf 'Output directory: %s\n' "${OUTPUT_DIR}"

exec "${PYTHON_BIN}" "${PYTHON_SCRIPT}" \
    --residual-csv "${RESIDUAL_CSV}" \
    --crop-config-csv "${CROP_CONFIG_CSV}" \
    --admin1-vector "${ADMIN1_VECTOR}" \
    --yield-loss-dir "${YIELD_LOSS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --overwrite \
    "$@"
