#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

unset PROJ_DATA
unset PROJ_LIB
unset PYTHONPATH

PYTHON="$PROJECT_DIR/.venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/IWR_anomaly_script/compute_iwr_anomalies_gamma.py"
DEFAULT_CONFIG="$PROJECT_DIR/IWR_anomaly_script/config/iwr_anomaly_monthly_gamma_config.json"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Virtual environment is missing. Run: uv sync"
    exit 1
fi

if [[ $# -gt 0 ]] && [[ "$1" != -* ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="$DEFAULT_CONFIG"
fi

echo "Running IWR gamma anomaly workflow with config: $CONFIG_PATH"
"$PYTHON" "$SCRIPT_PATH" "$CONFIG_PATH" "$@"

echo "Done."