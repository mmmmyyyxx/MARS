from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mars_core.logging_utils import write_csv

from .common import resolve_task_ids, run_task_method


def run_convergence_suite(
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
        task_curve_rows = []
        for method in suite_config["methods"]:
            row = run_task_method(
                suite_name="convergence",
                task=task,
                method=method,
                method_config=methods[method],
                settings=settings,
                model_config=model_config,
                run_dir=run_dir,
                prompt_loader=prompt_loader,
            )
            rows.append(row)
            history_path = (
                run_dir / "methods" / method / task_id / "prompt_accuracy_history.csv"
            )
            if history_path.exists():
                history = pd.read_csv(history_path).to_dict("records")
                for item in history:
                    task_curve_rows.append(
                        {
                            "task_id": task_id,
                            "method": method,
                            "iteration": item.get("iteration"),
                            "accuracy": item.get("accuracy"),
                        }
                    )
        write_csv(
            run_dir / "convergence" / f"{task_id}_curves.csv",
            task_curve_rows,
            ["task_id", "method", "iteration", "accuracy"],
        )
    return rows
