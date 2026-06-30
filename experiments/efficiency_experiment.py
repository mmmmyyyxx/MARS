from __future__ import annotations

from pathlib import Path
from typing import Any

from mars_core.logging_utils import write_csv

from .common import resolve_task_ids, run_task_method


def run_efficiency_suite(
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
                    suite_name="efficiency",
                    task=task,
                    method=method,
                    method_config=methods[method],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                )
            )
    columns = [
        "task_id",
        "method",
        "accuracy",
        "runtime_seconds",
        "api_calls",
        "tokens_prompt",
        "tokens_completion",
        "tokens_total",
        "cost_estimate",
        "num_iterations",
    ]
    write_csv(run_dir / "efficiency" / "efficiency_points.csv", rows, columns)
    return rows
