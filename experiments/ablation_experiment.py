from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import resolve_task_ids, run_task_method, write_table_pair


def run_ablation_suite(
    *,
    tasks,
    methods,
    suite_config: dict[str, Any],
    settings,
    model_config,
    run_dir: Path,
    prompt_loader,
) -> list[dict[str, Any]]:
    rows = []
    for task_id in resolve_task_ids(suite_config["tasks"], tasks):
        task = tasks[task_id]
        for method in suite_config["methods"]:
            rows.append(
                run_task_method(
                    suite_name="ablation",
                    task=task,
                    method=method,
                    method_config=methods[method],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                )
            )

    mars_by_task = {
        row["task_id"]: row.get("accuracy", 0.0)
        for row in rows
        if row.get("paper_method_id") == "mars"
    }
    for row in rows:
        row["delta"] = row.get("accuracy", 0.0) - mars_by_task.get(
            row.get("task_id"), 0.0
        )
    columns = [
        "task_id",
        "display_name",
        "method",
        "accuracy",
        "delta",
        "num_samples",
        "num_correct",
        "num_failed",
        "api_errors",
        "parse_errors",
        "runtime_seconds",
        "tokens_total",
        "cost_estimate",
    ]
    write_table_pair(run_dir / "tables" / "table3_ablation", rows, columns)
    return rows
