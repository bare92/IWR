import os
import re
import argparse
import logging
import sys
import json
from typing import Any

TAGS_JSON = os.getenv("TAGS")
DS_JSON   = os.getenv("DATASETS")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


# ---------------------------------------------------------------------------
# Config loading & merging
# ---------------------------------------------------------------------------

def _merge_dict(base: dict, other: dict) -> dict:
    for key, value in other.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _load_and_merge_configs(json_files: list[str]) -> dict:
    merged: dict = {}
    for file in json_files:
        with open(file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Hoist WORKFLOW keys to the top level so PROCESS is reachable at root
        if isinstance(data, dict) and isinstance(data.get("WORKFLOW"), dict):
            workflow = data.pop("WORKFLOW")
            data = {**data, **workflow}

        if not isinstance(data, dict):
            raise ValueError(f"Config file '{file}' must contain a JSON object")

        _merge_dict(merged, data)

    return merged


# ---------------------------------------------------------------------------
# Template variable resolution  {KEY} / {KEY.subkey}
# ---------------------------------------------------------------------------

def _resolve_str(value: str, context: dict, *, max_passes: int = 10) -> str:
    """Replace {KEY} and {KEY.subkey} placeholders using *context*."""
    for _ in range(max_passes):
        prev = value

        def replacer(match: re.Match) -> str:
            path = match.group(1).split(".")
            obj: Any = context
            for part in path:
                if isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                elif obj is context and len(path) == 1 and part in os.environ:
                    obj = os.environ[part]
                else:
                    return match.group(0)  # leave unresolved
            return str(obj) if not isinstance(obj, (dict, list)) else match.group(0)

        value = re.sub(r"\{([^}]+)\}", replacer, value)
        if value == prev:
            break  # fully resolved or stuck

    return value


def _resolve(obj: Any, context: dict) -> Any:
    if isinstance(obj, str):
        return _resolve_str(obj, context)
    if isinstance(obj, dict):
        return {k: _resolve(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(item, context) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Dataset path flattening
# ---------------------------------------------------------------------------

def _flatten_datasets(datasets: dict) -> dict:
    """
    Turn the nested DATASETS structure into a flat {name: full_path} mapping.

    Expected shape:
        DATASETS:
          <category>:
            path: <base_path>
            items:
              <name>:
                filename: <filename>
    """
    flat: dict[str, str] = {}
    for cat_data in datasets.values():
        if not isinstance(cat_data, dict):
            continue
        base = cat_data.get("path", "")
        for name, item in cat_data.get("items", {}).items():
            filename = item.get("filename", "")
            flat[name] = os.path.join(base, filename) if base and filename else filename
    return flat


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------

def _run_workflow(process_steps: list[dict]) -> None:
    from iwr_processing import preprocessing_functions as fn_module

    for step in process_steps:
        step_name = step.get("name", "unnamed")
        step_output = step.get("output")
        logger.info(f"=== Step: {step_name} ===")
        logger.info(f"    input  : {step.get('input', '-')}")
        logger.info(f"    output : {step_output or '-'}")

        tasks = step.get("process_list", [])
        pipe = None  # result forwarded from the previous function in the chain

        for i, task in enumerate(tasks):
            func_name = task.get("function")
            if not func_name:
                logger.warning(f"  Task in '{step_name}' has no 'function' key – skipping.")
                continue

            func = getattr(fn_module, func_name, None)
            if func is None:
                raise AttributeError(
                    f"Function '{func_name}' not found in iwr_processing.preprocessing_functions"
                )

            kwargs = {k: v for k, v in task.items() if k != "function"}

            # Forward piped outputs from the previous function as 'inputs'
            # unless this task already defines its own inputs.
            if pipe is not None and "inputs" not in kwargs:
                kwargs["inputs"] = pipe

            # Inject the step-level output path into the last function.
            is_last = i == len(tasks) - 1
            if is_last and step_output and "output" not in kwargs:
                kwargs["output"] = step_output

            logger.info(f"  [{func_name}] {list(kwargs.keys())}")
            pipe = func(**kwargs)

        logger.info(f"=== Done : {step_name} ===\n")


def _build_model(model_cfg: dict) -> None:
    from iwr_processing.iwr_core_process import IWRModel

    inputs = model_cfg.get("inputs", {}) if isinstance(model_cfg, dict) else {}
    land_cover_path = inputs.get("land_cover_fractional")
    soil_path = inputs.get("soil_type")
    crop_parameter_csv = inputs.get("crop_parameter_csv")

    if not land_cover_path or not soil_path:
        raise ValueError(
            "MODEL.inputs must define both 'land_cover_fractional' and 'soil_type'."
        )

    time_range_cfg = model_cfg.get("time_range")
    if isinstance(time_range_cfg, dict):
        start = time_range_cfg.get("start")
        end = time_range_cfg.get("end")
        if start or end:
            time_range = f"{start or ''}/{end or ''}"
        else:
            time_range = None
    elif time_range_cfg is None:
        time_range = None
    else:
        time_range = str(time_range_cfg)

    logger.info("Building IWR model with static layers...")
    model = IWRModel(
        time_range=time_range,
        land_cover_path=land_cover_path,
        soil_path=soil_path,
        crop_parameter_csv=crop_parameter_csv,
    )

    debug_cfg = model_cfg.get("debug", {}) if isinstance(model_cfg, dict) else {}
    if debug_cfg.get("print_summary", True):
        logger.info("Model summary: %s", model.summary())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an IWR processing workflow defined in JSON config files."
    )
    parser.add_argument(
        "options",
        nargs="+",
        help=(
            "One or more JSON config files and/or KEY=VALUE env overrides. "
            "Env overrides are applied before configs are loaded."
        ),
    )
    args = parser.parse_args()

    json_files: list[str] = []
    for token in args.options:
        if "=" in token:
            key, value = token.split("=", 1)
            os.environ[key] = value
            logger.info(f"ENV override: {key}={value}")
        elif token.endswith(".json"):
            json_files.append(token)
        else:
            logger.warning(f"Ignoring unrecognised argument: {token}")

    # Append paths provided via environment variables
    if TAGS_JSON:
        json_files.append(TAGS_JSON)
    if DS_JSON:
        json_files.append(DS_JSON)

    valid_files = [f for f in json_files if os.path.exists(f)]
    for f in set(json_files) - set(valid_files):
        logger.warning(f"Config file not found, skipping: {f}")

    if not valid_files:
        logger.error("No valid JSON config files were provided.")
        raise SystemExit(2)

    # ------------------------------------------------------------------ #
    # 1. Load & merge raw configs
    # ------------------------------------------------------------------ #
    config = _load_and_merge_configs(valid_files)

    # ------------------------------------------------------------------ #
    # 2. Build resolution context
    #    Pass 1 – resolve TAGS internal references ({Domain} inside paths)
    #    Pass 2 – resolve DATASETS paths using resolved TAGS
    #    Pass 3 – flatten DATASETS to {name: full_path}
    # ------------------------------------------------------------------ #
    tags: dict = config.get("TAGS", {})
    resolved_tags: dict = _resolve(tags, tags)          # e.g. DATA_PATH = .../DAO

    # Merge resolved tags into a working context
    context: dict = {**config, **resolved_tags}

    # Resolve DATASETS category paths
    datasets_resolved: dict = _resolve(config.get("DATASETS", {}), context)

    # Flatten to {name -> path} and re-resolve any remaining placeholders
    ds_flat: dict = _resolve(_flatten_datasets(datasets_resolved), context)

    logger.info("Resolved dataset paths:")
    for name, path in ds_flat.items():
        logger.info(f"  {name}: {path}")

    # ------------------------------------------------------------------ #
    # 3. Build final context used for process-step resolution
    # ------------------------------------------------------------------ #
    final_context: dict = {**context, "DATASETS": ds_flat}

    # ------------------------------------------------------------------ #
    # 4. Resolve PROCESS steps
    # ------------------------------------------------------------------ #
    process_steps: list = _resolve(config.get("PROCESS", []), final_context)

    model_cfg = _resolve(config.get("MODEL", {}), final_context)

    if process_steps:
        logger.info(f"Running {len(process_steps)} process step(s).\n")

        # ------------------------------------------------------------------ #
        # 5. Execute preprocessing workflow
        # ------------------------------------------------------------------ #
        _run_workflow(process_steps)
    else:
        if model_cfg:
            logger.info("No PROCESS steps found. Running MODEL only.")
        else:
            logger.warning("No PROCESS steps found in the merged config.")
            return

    # ------------------------------------------------------------------ #
    # 6. Optional model loading (for debugging static model inputs)
    # ------------------------------------------------------------------ #
    if model_cfg:
        _build_model(model_cfg)


if __name__ == "__main__":
    main()
    os._exit(0)
