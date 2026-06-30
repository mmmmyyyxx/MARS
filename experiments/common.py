from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

from baselines import ape, cot_few_shot, cot_zero_shot, opro, origin, pe2, protegi
from mars_core.api_client import API_CALL_COLUMNS, LLMClient
from mars_core.cache import DiskCache
from mars_core.logging_utils import append_jsonl, write_csv, write_json, write_text
from mars_core.mars_runner import (
    RunSettings,
    TaskSpec,
    evaluate_prompt,
    hash_rows,
    load_dataset,
    method_table_row,
    split_dataset,
    split_info,
    write_method_outputs,
)
from mars_core.mars_variants import run_mars_variant
from mars_core.official_mars_runner import run_official_mars
from mars_core.prompt_loader import PromptLoader
from mars_core.run_state import (
    build_run_state,
    expected_sample_ids,
    load_run_state,
    method_output_status,
    prompt_hash,
    read_predictions,
    should_skip_completed,
    stable_hash,
    write_run_state,
)
from mars_core.evaluator import PREDICTION_FIELDS

BASELINE_RUNNERS = {
    "origin": origin.run,
    "cot_zs": cot_zero_shot.run,
    "cot_fs": cot_few_shot.run,
    "ape": ape.run,
    "protegi": protegi.run,
    "opro": opro.run,
    "pe2": pe2.run,
}


def resolve_task_ids(selector: Any, tasks: dict[str, TaskSpec]) -> list[str]:
    if selector == "all":
        return list(tasks.keys())
    return list(selector)


def make_client(
    *,
    settings: RunSettings,
    model_config: dict[str, Any],
    cache_dir: Path,
    model: str | None = None,
    run_id: str = "",
    suite: str = "",
    method_id: str = "",
    pricing: dict[str, Any] | None = None,
) -> LLMClient:
    defaults = model_config.get("default", {})
    return LLMClient(
        model=model or settings.model,
        temperature=settings.temperature,
        api_key_env=defaults.get("api_key_env", "OPENAI_API_KEY"),
        base_url_env=defaults.get("base_url_env", "OPENAI_BASE_URL"),
        request_timeout=float(defaults.get("request_timeout", 120)),
        max_api_retries=int(defaults.get("max_api_retries", 5)),
        retry_backoff=defaults.get("retry_backoff", "exponential"),
        retry_delay=float(defaults.get("retry_delay", 5)),
        cache=DiskCache(cache_dir, enabled=settings.cache_enabled),
        dry_run=settings.dry_run,
        run_id=run_id,
        suite=suite,
        method_id=method_id,
        pricing=pricing,
    )


def _copy_to_legacy_layout(
    method_dir: Path, suite_name: str, method: str, task_id: str
) -> None:
    legacy_dir = method_dir.parents[2] / suite_name / method / task_id
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "predictions.csv",
        "best_prompt.txt",
        "final_prompt.txt",
        "diagnostics.md",
    ]:
        source = method_dir / filename
        if source.exists():
            shutil.copy2(source, legacy_dir / filename)


def _split_summary(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "num_opt_samples": len(splits.get("opt", [])),
        "num_val_samples": len(splits.get("val", [])),
        "num_test_samples": len(splits.get("test", [])),
        "opt_hash": hash_rows(splits.get("opt", [])),
        "val_hash": hash_rows(splits.get("val", [])),
        "test_hash": hash_rows(splits.get("test", [])),
    }


def _method_prompt_hash(method: str, prompts) -> str:
    if method == "origin":
        return prompt_hash(prompts.origin)
    if method == "cot_zs":
        return prompt_hash(prompts.cot_zero_shot)
    if method == "cot_fs":
        return prompt_hash(prompts.cot_few_shot)
    return prompt_hash(prompts.origin)


def _empty_api_log(method_dir: Path) -> None:
    write_csv(method_dir / "api_calls.csv", [], API_CALL_COLUMNS)


def _read_metrics(method_dir: Path) -> dict[str, Any]:
    path = method_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_history(method_dir: Path) -> list[dict[str, Any]]:
    path = method_dir / "prompt_accuracy_history.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _merge_predictions(
    *,
    existing: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {
        str(row.get("sample_id")): row for row in existing if row.get("sample_id") != ""
    }
    by_id.update(
        {str(row.get("sample_id")): row for row in fresh if row.get("sample_id") != ""}
    )
    merged = []
    seen = set()
    for row in test_rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in by_id:
            merged.append(by_id[sample_id])
            seen.add(sample_id)
    for sample_id, row in by_id.items():
        if sample_id not in seen:
            merged.append(row)
    return merged


def _try_resume_partial_predictions(
    *,
    client: LLMClient,
    task: TaskSpec,
    method: str,
    method_config: dict[str, Any],
    method_dir: Path,
    test_rows: list[dict[str, Any]],
    expected_ids: list[Any],
    config_hash: str,
    settings: RunSettings,
) -> dict[str, Any] | None:
    if not settings.resume or settings.force_rerun:
        return None
    status = method_output_status(method_dir, expected_ids)
    if status["status"] != "partial" or not status["missing_sample_ids"]:
        return None
    best_prompt_path = method_dir / "best_prompt.txt"
    predictions_path = method_dir / "predictions.csv"
    if not best_prompt_path.exists() or not predictions_path.exists():
        return None
    state = load_run_state(method_dir / "run_state.json")
    if (
        state
        and state.get("config_hash")
        and state.get("config_hash") != config_hash
        and not settings.reuse_compatible_cache
    ):
        return None
    missing = set(status["missing_sample_ids"])
    missing_rows = [row for row in test_rows if str(row.get("sample_id")) in missing]
    if not missing_rows:
        return None
    best_prompt = best_prompt_path.read_text(encoding="utf-8")
    final_prompt_path = method_dir / "final_prompt.txt"
    final_prompt = (
        final_prompt_path.read_text(encoding="utf-8")
        if final_prompt_path.exists()
        else best_prompt
    )
    history = _read_history(method_dir) or [
        {
            "iteration": 1,
            "prompt": best_prompt,
            "accuracy": "",
            "num_samples": "",
            "num_correct": "",
            "num_failed": "",
        }
    ]
    fresh = evaluate_prompt(
        client=client,
        task=task,
        rows=missing_rows,
        prompt=best_prompt,
        method=method,
        iteration=len(history) or 1,
    )
    existing = read_predictions(predictions_path)
    merged = _merge_predictions(existing=existing, fresh=fresh, test_rows=test_rows)
    metrics = write_method_outputs(
        out_dir=method_dir,
        task=task,
        method=method,
        method_config=method_config,
        predictions=merged,
        history=history,
        best_prompt=best_prompt,
        final_prompt=final_prompt,
        raw_logs="resumed_partial_predictions: true\n",
    )
    return metrics


def run_task_method(
    *,
    suite_name: str,
    task: TaskSpec,
    method: str,
    method_config: dict[str, Any],
    settings: RunSettings,
    model_config: dict[str, Any],
    run_dir: Path,
    prompt_loader: PromptLoader,
    model: str | None = None,
) -> dict[str, Any]:
    method_dir = run_dir / "methods" / method / task.task_id
    prompts = prompt_loader.load(task.task_id)
    all_rows = load_dataset(task.dataset_path, settings.max_samples)
    splits = split_dataset(all_rows, settings.eval_protocol, settings.split_seed)
    split_row = _split_summary(splits)
    method_type = method_config.get("type", "")
    effective_model = model or settings.model
    config_hash = stable_hash(
        {
            "suite": suite_name,
            "method": method,
            "method_config": method_config,
            "model": effective_model,
            "temperature": settings.temperature,
            "max_samples": settings.max_samples,
            "max_iterations": settings.max_iterations,
            "eval_protocol": settings.eval_protocol,
            "split_seed": settings.split_seed,
        }
    )
    expected_ids = expected_sample_ids(splits["test"])
    if should_skip_completed(
        method_dir=method_dir,
        expected_ids=expected_ids,
        config_hash=config_hash,
        force_rerun=settings.force_rerun,
        skip_existing=settings.skip_existing,
        resume=settings.resume,
        reuse_compatible_cache=settings.reuse_compatible_cache,
    ):
        status = method_output_status(method_dir, expected_ids)
        metrics = _read_metrics(method_dir)
        return {
            "skipped": True,
            "status": "skipped",
            "suite": suite_name,
            "task_id": task.task_id,
            "display_name": task.paper_display_name,
            "method": method_config.get("display_name", method),
            "method_id": method,
            "paper_method_id": method_config.get("paper_method_id", method),
            "model": effective_model,
            "eval_protocol": settings.eval_protocol,
            "exactness_level": method_config.get("exactness_level", ""),
            "exactness_note": method_config.get("exactness_note", ""),
            "accuracy": metrics.get("accuracy", ""),
            "num_samples": metrics.get("num_samples", status["num_prediction_rows"]),
            "num_correct": metrics.get("num_correct", ""),
            "num_failed": metrics.get("num_failed", ""),
            "api_errors": metrics.get("api_errors", ""),
            "parse_errors": metrics.get("parse_errors", ""),
            "num_iterations": metrics.get("num_iterations", ""),
            **split_row,
            **status,
        }
    task_dir = run_dir / "tasks" / task.task_id
    write_json(
        task_dir / "dataset_info.json",
        {"dataset_path": task.dataset_path, "num_loaded": len(all_rows)},
    )
    write_json(
        task_dir / "split_info.json",
        split_info(splits, settings.eval_protocol, settings.split_seed),
    )
    client = make_client(
        settings=settings,
        model_config=model_config,
        cache_dir=run_dir / ".cache",
        model=effective_model,
        run_id=run_dir.name,
        suite=suite_name,
        method_id=method,
        pricing=model_config.get("pricing", {}).get(effective_model, {}),
    )
    method_config = dict(method_config)
    method_config["model"] = effective_model
    method_config["temperature"] = settings.temperature
    method_config["pricing"] = model_config.get("pricing", {}).get(
        effective_model, {}
    )
    method_config["config_hash"] = config_hash
    method_config["eval_protocol"] = settings.eval_protocol
    method_config.update(split_row)

    start = time.time()
    metrics = _try_resume_partial_predictions(
        client=client,
        task=task,
        method=method,
        method_config=method_config,
        method_dir=method_dir,
        test_rows=splits["test"],
        expected_ids=expected_ids,
        config_hash=config_hash,
        settings=settings,
    )
    if metrics is not None:
        pass
    elif method in BASELINE_RUNNERS:
        runner = BASELINE_RUNNERS[method]
        if method in {"origin", "cot_zs", "cot_fs"}:
            metrics = runner(
                client=client,
                task=task,
                prompts=prompts,
                test_rows=splits["test"],
                method_config=method_config,
                out_dir=method_dir,
                few_shot_rows=splits["opt"],
            )
        else:
            metrics = runner(
                client=client,
                task=task,
                prompts=prompts,
                opt_rows=splits["opt"],
                val_rows=splits["val"],
                test_rows=splits["test"],
                method_config=method_config,
                out_dir=method_dir,
                max_iterations=settings.max_iterations,
            )
    elif method_type == "mars_official":
        metrics = run_official_mars(
            client=client,
            task=task,
            prompts=prompts,
            opt_rows=splits["opt"],
            val_rows=splits["val"],
            test_rows=splits["test"],
            method=method,
            method_config=method_config,
            out_dir=method_dir,
            max_iterations=settings.max_iterations,
            max_critic_revisions=settings.max_critic_revisions,
        )
    elif method_type == "mars_light" or method.startswith("mars"):
        metrics = run_mars_variant(
            client=client,
            task=task,
            prompts=prompts,
            opt_rows=splits["opt"],
            val_rows=splits["val"],
            test_rows=splits["test"],
            method=method,
            method_config=method_config,
            out_dir=method_dir,
            max_iterations=settings.max_iterations,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    for error in client.stats.error_records:
        append_jsonl(run_dir / "logs" / "api_errors.jsonl", error)
    if client.stats.call_records:
        write_csv(method_dir / "api_calls.csv", client.stats.call_records, API_CALL_COLUMNS)
    else:
        _empty_api_log(method_dir)

    runtime = time.time() - start
    metrics["runtime_seconds"] = runtime
    state = build_run_state(
        run_id=run_dir.name,
        suite=suite_name,
        method_id=method,
        task_id=task.task_id,
        model=effective_model,
        temperature=settings.temperature,
        max_samples=settings.max_samples,
        dataset_hash=hash_rows(all_rows),
        prompt_hash_value=_method_prompt_hash(method, prompts),
        config_hash=config_hash,
        expected_ids=expected_ids,
        predictions_path=method_dir / "predictions.csv",
        status="completed",
    )
    write_run_state(method_dir / "run_state.json", state)
    write_json(method_dir / "metrics.json", metrics)
    row = method_table_row(
        task=task,
        method=method,
        method_config=method_config,
        metrics=metrics,
        client=client,
        runtime_seconds=runtime,
    )
    row["suite"] = suite_name
    row["model"] = effective_model
    row["paper_method_id"] = method_config.get("paper_method_id", method)
    row["status"] = "completed"
    row["eval_protocol"] = settings.eval_protocol
    row.update(split_row)
    _copy_to_legacy_layout(method_dir, suite_name, method, task.task_id)
    return row


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append(
            "|" + "|".join(str(row.get(column, "")) for column in columns) + "|"
        )
    return "\n".join(lines) + "\n"


def write_table_pair(
    path_base: Path, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    from mars_core.logging_utils import write_csv

    write_csv(path_base.with_suffix(".csv"), rows, columns)
    write_text(path_base.with_suffix(".md"), markdown_table(rows, columns))
