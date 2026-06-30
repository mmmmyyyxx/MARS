from __future__ import annotations

from pathlib import Path
from typing import Any

from mars_core.api_client import LLMClient
from mars_core.cache import DiskCache
from mars_core.mars_runner import (
    evaluate_fixed_prompt_method,
    load_dataset,
    method_table_row,
    split_dataset,
)

from .common import resolve_task_ids, run_task_method, write_table_pair


def _target_client(settings, model_config, run_dir: Path, model: str) -> LLMClient:
    defaults = model_config.get("default", {})
    return LLMClient(
        model=model,
        temperature=settings.temperature,
        api_key_env=defaults.get("api_key_env", "OPENAI_API_KEY"),
        base_url_env=defaults.get("base_url_env", "OPENAI_BASE_URL"),
        request_timeout=float(defaults.get("request_timeout", 120)),
        max_api_retries=int(defaults.get("max_api_retries", 5)),
        retry_backoff=defaults.get("retry_backoff", "exponential"),
        retry_delay=float(defaults.get("retry_delay", 5)),
        cache=DiskCache(run_dir / ".cache", enabled=settings.cache_enabled),
        dry_run=settings.dry_run,
    )


def run_transfer_suite(
    *,
    tasks,
    methods,
    suite_config: dict[str, Any],
    settings,
    model_config,
    run_dir: Path,
    prompt_loader,
    source_model: str | None = None,
    target_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    source = source_model or settings.model
    targets = target_models or ["deepseek-r1", "gpt-3.5", "gpt-4", "gpt-4o"]
    for task_id in resolve_task_ids(suite_config["tasks"], tasks):
        task = tasks[task_id]
        rows.append(
            run_task_method(
                suite_name="transfer_source",
                task=task,
                method="mars",
                method_config=methods["mars"],
                settings=settings,
                model_config=model_config,
                run_dir=run_dir,
                prompt_loader=prompt_loader,
                model=source,
            )
        )
        source_best_prompt = (
            run_dir / "methods" / "mars" / task.task_id / "best_prompt.txt"
        ).read_text(encoding="utf-8")
        all_rows = load_dataset(task.dataset_path, settings.max_samples)
        splits = split_dataset(all_rows, settings.eval_protocol, settings.split_seed)
        for target in targets:
            for method in suite_config["methods"]:
                method_dir = (
                    run_dir / "transfer" / target / "methods" / method / task.task_id
                )
                prompts = prompt_loader.load(task.task_id)
                if method == "mars":
                    prompt = source_best_prompt
                elif method == "origin":
                    prompt = prompts.origin
                elif method == "cot_zs":
                    prompt = prompts.cot_zero_shot
                else:
                    continue
                client = _target_client(settings, model_config, run_dir, target)
                method_config = dict(methods[method])
                method_config["model"] = target
                method_config["transfer_target_evaluation_only"] = True
                metrics = evaluate_fixed_prompt_method(
                    client=client,
                    task=task,
                    prompt=prompt,
                    test_rows=splits["test"],
                    method=method,
                    method_config=method_config,
                    out_dir=method_dir,
                )
                row = method_table_row(
                    task=task,
                    method=method,
                    method_config=method_config,
                    metrics=metrics,
                    client=client,
                    runtime_seconds=metrics.get("runtime_seconds", 0.0),
                )
                row["suite"] = "transfer"
                row["model"] = target
                row["target_model"] = target
                rows.append(row)
    columns = [
        "target_model",
        "task_id",
        "display_name",
        "method",
        "accuracy",
        "num_samples",
        "num_correct",
        "num_failed",
        "api_errors",
        "parse_errors",
        "runtime_seconds",
        "tokens_total",
        "cost_estimate",
    ]
    write_table_pair(
        run_dir / "tables" / "table4_transfer",
        [row for row in rows if row.get("target_model")],
        columns,
    )
    return rows
