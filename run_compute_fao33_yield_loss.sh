#!/usr/bin/env bash

# Launch the FAO-33 seasonal yield-loss post-processing step.
#
# Put this file in either:
#   ~/workspaces/RB/IWR/
# or:
#   ~/workspaces/RB/IWR/scripts_additional/
#
# It uses the currently active Python/Conda environment.

set -Eeuo pipefail

launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${launcher_dir}/scripts_additional/compute_fao33_yield_loss.py" ]] \
    && [[ -f "${launcher_dir}/IWR_scripts/config/config_eraL_theoretical_rainfed.json" ]]; then
    yield_script="${launcher_dir}/scripts_additional/compute_fao33_yield_loss.py"
    config="${launcher_dir}/IWR_scripts/config/config_eraL_theoretical_rainfed.json"
elif [[ -f "${launcher_dir}/compute_fao33_yield_loss.py" ]] \
    && [[ -f "${launcher_dir}/../IWR_scripts/config/config_eraL_theoretical_rainfed.json" ]]; then
    project_root="$(cd -- "${launcher_dir}/.." && pwd)"
    yield_script="${launcher_dir}/compute_fao33_yield_loss.py"
    config="${project_root}/IWR_scripts/config/config_eraL_theoretical_rainfed.json"
else
    echo "ERROR: Could not locate the FAO-33 script and IWR configuration." >&2
    echo "Place this launcher in the IWR project root or scripts_additional directory." >&2
    exit 1
fi

python_bin="${PYTHON_BIN:-python}"
start_year="${START_YEAR:-1993}"
end_year="${END_YEAR:-2025}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${python_bin}" >&2
    exit 1
fi

echo "Running FAO-33 yield-loss calculation for SOS years ${start_year}-${end_year}"

exec "${python_bin}" "${yield_script}" \
    --config "${config}" \
    --start-year "${start_year}" \
    --end-year "${end_year}" \
    --overwrite \
    "$@"
