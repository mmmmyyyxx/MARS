import csv
from datetime import datetime
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import yaml


SUMMARY_FIELDS = [
    "task_id",
    "group",
    "dataset_path",
    "question_type",
    "status",
    "num_samples",
    "num_success",
    "num_failed",
    "best_accuracy",
    "final_accuracy",
    "best_iteration",
    "stopped_reason",
    "total_runtime_seconds",
]


def make_run_dir(output_dir: str) -> str:
    run_dir = os.path.join(output_dir, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tasks"), exist_ok=True)
    return run_dir


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def write_prompt_history(path: str, history: Iterable[Tuple[str, float]], stopped_reason: str) -> None:
    rows = []
    previous = None
    history_list = list(history)
    for index, (prompt, accuracy) in enumerate(history_list, start=1):
        delta = "" if previous is None else accuracy - previous
        rows.append({
            "iteration": index,
            "accuracy": accuracy,
            "delta_accuracy": delta,
            "prompt": prompt,
            "stopped_reason": stopped_reason if index == len(history_list) else "",
        })
        previous = accuracy
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "accuracy", "delta_accuracy", "prompt", "stopped_reason"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(run_dir: str, rows: List[Dict[str, Any]]) -> None:
    with open(os.path.join(run_dir, "summary.csv"), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def write_report(run_dir: str, rows: List[Dict[str, Any]], dry_run: bool) -> None:
    succeeded = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") != "success"]
    lines = [
        "# MARS Reproduction Report",
        "",
        "This reproduction script runs only the MARS method. It does not reproduce baseline comparisons, ablation studies, efficiency comparisons, or cross-model transfer experiments.",
        "",
        f"Dry run: {dry_run}",
        f"Total tasks: {len(rows)}",
        f"Successful tasks: {len(succeeded)}",
        f"Failed tasks: {len(failed)}",
        "",
        "## Successful Tasks",
    ]
    if succeeded:
        for row in succeeded:
            lines.append(f"- {row['task_id']}: final_accuracy={row['final_accuracy']}, best_accuracy={row['best_accuracy']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Failed Tasks"])
    if failed:
        for row in failed:
            lines.append(f"- {row['task_id']}: {row['stopped_reason']}")
    else:
        lines.append("- None")
    write_text(os.path.join(run_dir, "report.md"), "\n".join(lines) + "\n")
