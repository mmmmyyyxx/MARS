from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from baselines import ape, cot_few_shot, cot_zero_shot, opro, origin, pe2, protegi
from mars_core.api_client import LLMClient
from mars_core.cache import DiskCache
from mars_core.logging_utils import append_jsonl, write_json, write_text
from mars_core.mars_runner import (
    RunSettings,
    TaskSpec,
    load_dataset,
    method_table_row,
    split_dataset,
    split_info,
)
from mars_core.mars_variants import run_mars_variant
from mars_core.prompt_loader import PromptLoader

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
    predictions_path = method_dir / "predictions.csv"
    if (
        settings.skip_existing
        and predictions_path.exists()
        and not settings.force_rerun
    ):
        return {"skipped": True, "task_id": task.task_id, "method_id": method}

    prompts = prompt_loader.load(task.task_id)
    all_rows = load_dataset(task.dataset_path, settings.max_samples)
    splits = split_dataset(all_rows, settings.eval_protocol, settings.split_seed)
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
        model=model,
    )
    method_config = dict(method_config)
    method_config["model"] = model or settings.model
    method_config["temperature"] = settings.temperature
    method_config["pricing"] = model_config.get("pricing", {}).get(
        model or settings.model, {}
    )

    start = time.time()
    if method in BASELINE_RUNNERS:
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
    elif method.startswith("mars"):
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

    runtime = time.time() - start
    row = method_table_row(
        task=task,
        method=method,
        method_config=method_config,
        metrics=metrics,
        client=client,
        runtime_seconds=runtime,
    )
    row["suite"] = suite_name
    row["model"] = model or settings.model
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
