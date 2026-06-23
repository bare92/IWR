import os
import re
import argparse
import logging
import sys
import json
from datetime import date
from typing import Any

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
                folder: <optional_folder>
                filename: <filename>
    """
    flat: dict[str, str] = {}
    for cat_data in datasets.values():
        if not isinstance(cat_data, dict):
            continue
        base = cat_data.get("path", "")
        for name, item in cat_data.get("items", {}).items():
            folder = item.get("folder", "")
            filename = item.get("filename", "")
            if filename:
                parts = [part for part in (base, folder, filename) if part]
                flat[name] = os.path.join(*parts) if parts else filename
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
    from iwr_processing.crop_calendar import load_crop_calendars_from_csv
    from iwr_processing.monthly_iwr_runner import MonthlyIWRRunner
    from iwr_processing.netcdf_forcing_reader import (
        MonthlyNetCDFForcingReader,
        MonthlyNetCDFVariableSpec,
    )
    import rasterio

    def _parse_iso_date(value: str | None, label: str) -> date:
        if not value:
            raise ValueError(f"MODEL.time_range.{label} is required for monthly runner.")
        return date.fromisoformat(str(value))

    def _resolve_crop_calendar(
        calendars_by_key: dict[tuple[str, int], Any],
        crop_name: str,
        target_year: int,
    ) -> Any:
        direct = calendars_by_key.get((crop_name, target_year))
        if direct is not None:
            return direct

        same_crop = [cal for (cid, _), cal in calendars_by_key.items() if cid == crop_name]
        if same_crop:
            same_crop.sort(key=lambda cal: abs(cal.year - target_year))
            return same_crop[0]

        available = sorted({cid for (cid, _) in calendars_by_key.keys()})
        raise KeyError(
            f"No crop calendar found for crop '{crop_name}'. Available crop_ids: {available}"
        )

    inputs = (
        model_cfg.get("static_inputs")
        or model_cfg.get("inputs")
        or {}
    ) if isinstance(model_cfg, dict) else {}
    land_cover_path = inputs.get("land_cover_fractional")
    soil_path = inputs.get("soil_type")
    crop_parameter_csv = (
        inputs.get("crop_parameter_csv")
        or model_cfg.get("crop_parameter_csv")
        or None
    )
    initial_saturation = inputs.get("initial_saturation")
    initial_condition = inputs.get("initial_condition", "watneeds_half_taw")

    processing_cfg = model_cfg.get("processing", {}) if isinstance(model_cfg, dict) else {}
    peff_reduction_pct = processing_cfg.get("peff_reduction_pct", inputs.get("peff_reduction_pct", 5.0))

    soil_parameters = model_cfg.get("soil_parameters", {}) if isinstance(model_cfg, dict) else {}
    taw_layer = soil_parameters.get("taw_layer")
    smax_layer = soil_parameters.get("smax_layer")
    fmax_layer = soil_parameters.get("fmax_layer")

    if not land_cover_path or not soil_path:
        raise ValueError(
            "MODEL.static_inputs (or MODEL.inputs) must define both "
            "'land_cover_fractional' and 'soil_type'."
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
        initial_condition=initial_condition,
        initial_saturation=initial_saturation,
        peff_reduction_pct=peff_reduction_pct,
        taw_layer=taw_layer,
        smax_layer=smax_layer,
        fmax_layer=fmax_layer,
    )

    debug_cfg = model_cfg.get("debug", {}) if isinstance(model_cfg, dict) else {}
    if debug_cfg.get("print_summary", True):
        logger.info("Model summary: %s", model.summary())

    runner_cfg = model_cfg.get("runner", {}) if isinstance(model_cfg, dict) else {}
    if runner_cfg.get("type") != "monthly_iwr_runner":
        return

    logger.info("Runner type is monthly_iwr_runner. Starting streamed simulation...")

    if not isinstance(time_range_cfg, dict):
        raise ValueError("MODEL.time_range must be an object with 'start' and 'end' for monthly runner.")

    start_date = _parse_iso_date(time_range_cfg.get("start"), "start")
    end_date = _parse_iso_date(time_range_cfg.get("end"), "end")

    forcing_cfg = model_cfg.get("forcing", {}) if isinstance(model_cfg, dict) else {}
    root_dir = forcing_cfg.get("root_dir")
    if not root_dir:
        raise ValueError("MODEL.forcing.root_dir is required for monthly runner.")

    variable_cfg = forcing_cfg.get("variables", {})
    if variable_cfg:
        variable_specs: dict[str, MonthlyNetCDFVariableSpec] = {}
        for var_name, spec in variable_cfg.items():
            variable_specs[var_name] = MonthlyNetCDFVariableSpec(
                name=var_name,
                folder=spec.get("folder", var_name),
                file_pattern=spec.get("file_pattern", "{variable}_{year}_{month:02d}.nc"),
                variable_name=spec.get("variable_name"),
                units=spec.get("units"),
            )
        forcing_reader = MonthlyNetCDFForcingReader(
            root_dir=root_dir,
            variable_specs=variable_specs,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        forcing_reader = MonthlyNetCDFForcingReader.from_dao_layout(
            root_dir=root_dir,
            start_date=start_date,
            end_date=end_date,
        )

    crop_calendars_cfg = model_cfg.get("crop_calendars", {}) if isinstance(model_cfg, dict) else {}
    crop_calendar_path = crop_calendars_cfg.get("csv_path")
    if not crop_calendar_path:
        raise ValueError(
            "MODEL.crop_calendars.csv_path is required for monthly_iwr_runner."
        )

    simulation_years = list(range(start_date.year, end_date.year + 1))
    calendars_by_key = load_crop_calendars_from_csv(
        crop_calendar_path,
        years=simulation_years,
    )
    crop_name = runner_cfg.get("crop_name")
    if not crop_name:
        raise ValueError("MODEL.runner.crop_name is required for monthly_iwr_runner.")
    crop_calendar = _resolve_crop_calendar(calendars_by_key, crop_name, start_date.year)

    output_dir = runner_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("MODEL.runner.output_dir is required for monthly_iwr_runner.")

    use_phenology_kc = bool(runner_cfg.get("use_phenology_kc", False))
    phenology_nodata_value = runner_cfg.get("phenology_nodata_value")
    skip_iwr_when_inactive = bool(runner_cfg.get("skip_iwr_when_inactive", False))

    phenology_layers: dict[str, Any] | None = None
    if use_phenology_kc:
        expected_shape = model.soil.field_capacity.shape

        def _load_phenology_layer(key: str) -> Any:
            path = inputs.get(key)
            if not path:
                raise ValueError(
                    f"MODEL.runner.use_phenology_kc=True requires MODEL.static_inputs.{key}."
                )

            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"Phenology layer '{key}' shape {arr.shape} does not match model grid {expected_shape}."
                    )
                nonlocal phenology_nodata_value
                if phenology_nodata_value is None and src.nodata is not None:
                    phenology_nodata_value = src.nodata
            return arr

        phenology_layers = {
            "phenoe1": _load_phenology_layer("phenoe1"),
            "phenom1": _load_phenology_layer("phenom1"),
            "phenos1": _load_phenology_layer("phenos1"),
            "phenosen1": _load_phenology_layer("phenosen1"),
        }

        season2_keys = ("phenoe2", "phenom2", "phenos2", "phenosen2")
        present_s2 = [k for k in season2_keys if inputs.get(k)]
        if present_s2 and len(present_s2) != len(season2_keys):
            raise ValueError(
                "If any season-2 phenology layer is provided, all must be provided: "
                f"{season2_keys}. Present: {present_s2}"
            )
        if len(present_s2) == len(season2_keys):
            for key in season2_keys:
                phenology_layers[key] = _load_phenology_layer(key)

    monthly_runner = MonthlyIWRRunner(
        model=model,
        forcing_reader=forcing_reader,
        crop_calendar=crop_calendar,
        crop_name=crop_name,
        output_dir=output_dir,
        irrigated_mask=inputs.get("irrigated_mask"),
        use_direct_etc=bool(runner_cfg.get("use_direct_etc", False)),
        write_monthly_outputs=bool(runner_cfg.get("write_monthly_outputs", True)),
        phenology_layers=phenology_layers,
        phenology_nodata_value=phenology_nodata_value,
        use_phenology_kc=use_phenology_kc,
        skip_iwr_when_inactive=skip_iwr_when_inactive,
    )

    result = monthly_runner.run()
    logger.info(
        "Monthly runner complete. Files written: %d. Final soil storage shape: %s",
        len(result.written_files),
        result.final_soil_storage_mm.shape,
    )


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

    # Append paths provided via environment variables.
    # Read them here so KEY=VALUE CLI overrides are honored.
    tags_json = os.getenv("TAGS")
    datasets_json = os.getenv("DATASETS")
    if tags_json:
        json_files.append(tags_json)
    if datasets_json:
        json_files.append(datasets_json)

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
    if "PROCESSED_DIR" not in context and "DATA_PATH" in resolved_tags:
        context["PROCESSED_DIR"] = os.path.join(resolved_tags["DATA_PATH"], "processed")

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
    datasets_context = dict(datasets_resolved)
    datasets_context.update(ds_flat)
    final_context: dict = {**context, "DATASETS": datasets_context}

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
    # Avoid forced interpreter exit under debugpy (can surface as non-zero exit codes).
    if sys.gettrace() is None:
        os._exit(0)
