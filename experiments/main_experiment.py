from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import resolve_task_ids, run_task_method, write_table_pair


def run_main_suite(
    *,
    tasks,
    methods,
    suite_config: dict[str, Any],
    settings,
    model_config,
    run_dir: Path,
    prompt_loader,
    selected_methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    task_ids = resolve_task_ids(suite_config["tasks"], tasks)
    method_ids = selected_methods or suite_config["methods"]
    rows = []
    for task_id in task_ids:
        task = tasks[task_id]
        for method in method_ids:
            rows.append(
                run_task_method(
                    suite_name="main",
                    task=task,
                    method=method,
                    method_config=methods[method],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                )
            )

    table_columns = [
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
        "tokens_prompt",
        "tokens_completion",
        "tokens_total",
        "cost_estimate",
    ]
    general = [
        row
        for row in rows
        if row.get("task_id") and tasks[row["task_id"]].group == "BBH"
    ]
    domain = [
        row
        for row in rows
        if row.get("task_id") and tasks[row["task_id"]].group != "BBH"
    ]
    write_table_pair(run_dir / "tables" / "table1_general", general, table_columns)
    write_table_pair(run_dir / "tables" / "table2_domain", domain, table_columns)
    return rows
