#!/usr/bin/env bash

# Launch the seasonal ET accumulation from the IWR project.
#
# Put this file in either:
#   ~/workspaces/RB/IWR/
# or:
#   ~/workspaces/RB/IWR/scripts_additional/
#
# It uses the currently active Python/Conda environment.

set -Eeuo pipefail

launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${launcher_dir}/scripts_additional/accumulate_seasonal_et.py" ]]; then
    project_root="${launcher_dir}"
elif [[ -f "${launcher_dir}/accumulate_seasonal_et.py" ]] \
    && [[ -f "${launcher_dir}/../IWR_scripts/config/config_eraL_theoretical_rainfed.json" ]]; then
    project_root="$(cd -- "${launcher_dir}/.." && pwd)"
else
    echo "ERROR: Could not locate the IWR project files." >&2
    echo "Place this launcher in the IWR root or in its scripts_additional directory." >&2
    exit 1
fi

python_bin="${PYTHON_BIN:-python}"
start_year="${START_YEAR:-1993}"
end_year="${END_YEAR:-2025}"

accumulator="${project_root}/scripts_additional/accumulate_seasonal_et.py"
config="${project_root}/IWR_scripts/config/config_eraL_theoretical_rainfed.json"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${python_bin}" >&2
    exit 1
fi

if [[ ! -f "${accumulator}" ]]; then
    echo "ERROR: Accumulation script not found: ${accumulator}" >&2
    exit 1
fi

if [[ ! -f "${config}" ]]; then
    echo "ERROR: Configuration file not found: ${config}" >&2
    exit 1
fi

echo "Running seasonal ET accumulation for SOS years ${start_year}-${end_year}"

exec "${python_bin}" "${accumulator}" \
    --config "${config}" \
    --start-year "${start_year}" \
    --end-year "${end_year}" \
    --overwrite \
    "$@"
