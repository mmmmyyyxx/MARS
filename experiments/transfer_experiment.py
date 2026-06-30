from __future__ import annotations

from pathlib import Path
from typing import Any

from mars_core.api_client import API_CALL_COLUMNS, LLMClient
from mars_core.cache import DiskCache
from mars_core.logging_utils import write_csv, write_json
from mars_core.mars_runner import (
    evaluate_fixed_prompt_method,
    hash_rows,
    load_dataset,
    method_table_row,
    split_dataset,
)
from mars_core.run_state import (
    build_run_state,
    expected_sample_ids,
    prompt_hash,
    stable_hash,
)

from .common import resolve_task_ids, run_task_method, write_table_pair


def _target_client(
    settings, model_config, run_dir: Path, model: str, method_id: str
) -> LLMClient:
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
        run_id=run_dir.name,
        suite="transfer",
        method_id=method_id,
        pricing=model_config.get("pricing", {}).get(model, {}),
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
                method="mars_official",
                method_config=methods["mars_official"],
                settings=settings,
                model_config=model_config,
                run_dir=run_dir,
                prompt_loader=prompt_loader,
                model=source,
            )
        )
        source_best_prompt = (
            run_dir / "methods" / "mars_official" / task.task_id / "best_prompt.txt"
        ).read_text(encoding="utf-8")
        all_rows = load_dataset(task.dataset_path, settings.max_samples)
        splits = split_dataset(all_rows, settings.eval_protocol, settings.split_seed)
        for target in targets:
            for method in suite_config["methods"]:
                method_dir = (
                    run_dir / "transfer" / target / "methods" / method / task.task_id
                )
                prompts = prompt_loader.load(task.task_id)
                if method == "mars_official":
                    prompt = source_best_prompt
                elif method == "origin":
                    prompt = prompts.origin
                elif method == "cot_zs":
                    prompt = prompts.cot_zero_shot
                else:
                    continue
                method_config = dict(methods[method])
                method_config["model"] = target
                method_config["transfer_target_evaluation_only"] = True
                method_config["pricing"] = model_config.get("pricing", {}).get(
                    target, {}
                )
                method_config["eval_protocol"] = settings.eval_protocol
                method_config["num_opt_samples"] = len(splits["opt"])
                method_config["num_val_samples"] = len(splits["val"])
                method_config["num_test_samples"] = len(splits["test"])
                method_config["opt_hash"] = hash_rows(splits["opt"])
                method_config["val_hash"] = hash_rows(splits["val"])
                method_config["test_hash"] = hash_rows(splits["test"])
                method_config["config_hash"] = stable_hash(
                    {
                        "suite": "transfer",
                        "method": method,
                        "target_model": target,
                        "method_config": method_config,
                        "temperature": settings.temperature,
                        "max_samples": settings.max_samples,
                        "eval_protocol": settings.eval_protocol,
                        "split_seed": settings.split_seed,
                    }
                )
                client = _target_client(settings, model_config, run_dir, target, method)
                metrics = evaluate_fixed_prompt_method(
                    client=client,
                    task=task,
                    prompt=prompt,
                    test_rows=splits["test"],
                    method=method,
                    method_config=method_config,
                    out_dir=method_dir,
                )
                metrics["num_iterations"] = 0
                write_json(method_dir / "metrics.json", metrics)
                write_csv(
                    method_dir / "api_calls.csv",
                    client.stats.call_records,
                    API_CALL_COLUMNS,
                )
                state = build_run_state(
                    run_id=run_dir.name,
                    suite="transfer",
                    method_id=method,
                    task_id=task.task_id,
                    model=target,
                    temperature=settings.temperature,
                    max_samples=settings.max_samples,
                    dataset_hash=hash_rows(all_rows),
                    prompt_hash_value=prompt_hash(prompt),
                    config_hash=method_config["config_hash"],
                    expected_ids=expected_sample_ids(splits["test"]),
                    predictions_path=method_dir / "predictions.csv",
                    status="completed",
                )
                write_json(method_dir / "run_state.json", state)
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
                row["paper_method_id"] = method_config.get("paper_method_id", method)
                row["status"] = "completed"
                row["eval_protocol"] = settings.eval_protocol
                row["num_opt_samples"] = len(splits["opt"])
                row["num_val_samples"] = len(splits["val"])
                row["num_test_samples"] = len(splits["test"])
                row["opt_hash"] = hash_rows(splits["opt"])
                row["val_hash"] = hash_rows(splits["val"])
                row["test_hash"] = hash_rows(splits["test"])
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
