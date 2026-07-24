#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

unset PROJ_DATA
unset PROJ_LIB
unset PYTHONPATH

PYTHON="$PROJECT_DIR/.venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/IWR_scripts/iwr_simple_main.py"
DEFAULT_CONFIG="$PROJECT_DIR/IWR_scripts/config/config_eraL.json"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Virtual environment is missing. Run: uv sync"
    exit 1
fi

echo "Running simple IWR workflow with config: $CONFIG_PATH"
"$PYTHON" "$SCRIPT_PATH" "$CONFIG_PATH"

echo "Done."