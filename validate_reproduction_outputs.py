from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from mars_core.mars_runner import load_yaml
from mars_core.run_state import REQUIRED_METHOD_FILES


def latest_run(results_root: Path) -> Path:
    runs = sorted(path for path in results_root.glob("run_*") if path.is_dir())
    if not runs:
        raise SystemExit(f"No run_* directories found under {results_root}")
    return runs[-1]


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.latest:
        return latest_run(Path(args.results_root))
    if args.run_dir:
        return Path(args.run_dir)
    return latest_run(Path(args.results_root))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_main_pairs(run_dir: Path) -> set[tuple[str, str]]:
    tasks = load_yaml("configs/tasks.yaml")
    suites = load_yaml("configs/suites.yaml")
    main = suites.get("main", {})
    task_ids = list(tasks.keys()) if main.get("tasks") == "all" else list(main.get("tasks", []))
    method_ids = list(main.get("methods", []))
    summary = read_csv_rows(run_dir / "summary.csv")
    if summary:
        task_ids = sorted(
            {
                row["task_id"]
                for row in summary
                if row.get("suite") == "main" and row.get("task_id")
            }
        ) or task_ids
        method_ids = sorted(
            {
                row["method_id"]
                for row in summary
                if row.get("suite") == "main" and row.get("method_id")
            }
        ) or method_ids
    return {(task_id, method_id) for task_id in task_ids for method_id in method_ids}


def validate_method_dir(method_dir: Path) -> list[str]:
    issues = []
    for filename in REQUIRED_METHOD_FILES:
        if not (method_dir / filename).exists():
            issues.append(f"{method_dir}: missing {filename}")
    predictions = read_csv_rows(method_dir / "predictions.csv")
    seen = set()
    duplicates = set()
    parse_errors = 0
    for row in predictions:
        sample_id = row.get("sample_id", "")
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)
        if row.get("error_type") in {"parse_error", "empty_output", "invalid_answer_format"}:
            parse_errors += 1
    if duplicates:
        issues.append(f"{method_dir}: duplicate sample_id values {sorted(duplicates)}")
    if not predictions:
        issues.append(f"{method_dir}: predictions.csv has no rows")
    best_prompt = method_dir / "best_prompt.txt"
    if best_prompt.exists() and not best_prompt.read_text(encoding="utf-8").strip():
        issues.append(f"{method_dir}: best_prompt.txt is empty")
    api_calls = method_dir / "api_calls.csv"
    if api_calls.exists() and api_calls.stat().st_size == 0:
        issues.append(f"{method_dir}: api_calls.csv is empty/unparseable")
    state_path = method_dir / "run_state.json"
    if state_path.exists():
        state = read_json(state_path)
        expected = {str(item) for item in state.get("expected_sample_ids", [])}
        present = {str(row.get("sample_id")) for row in predictions}
        missing = sorted(expected - present)
        if missing:
            issues.append(f"{method_dir}: missing predictions for sample_ids {missing[:20]}")
    if parse_errors:
        issues.append(f"{method_dir}: parse_errors={parse_errors}")
    return issues


def validate_run(run_dir: Path) -> list[str]:
    issues = []
    required_run_files = [
        "run_config.yaml",
        "environment.json",
        "git_info.json",
        "preflight_report.csv",
        "summary.csv",
        "summary.json",
        "paper_comparison.csv",
        "coverage.json",
        "paper_reproduction_report.md",
        "final_report.md",
        "api_calls.csv",
        "token_summary.csv",
        "latency_summary.csv",
        "cost_summary.csv",
    ]
    for filename in required_run_files:
        if not (run_dir / filename).exists():
            issues.append(f"{run_dir}: missing {filename}")
    for dirname in ["tables", "figures", "methods", "tasks"]:
        if not (run_dir / dirname).exists():
            issues.append(f"{run_dir}: missing directory {dirname}")

    pairs = expected_main_pairs(run_dir)
    for task_id, method_id in sorted(pairs):
        method_dir = run_dir / "methods" / method_id / task_id
        if not method_dir.exists():
            issues.append(f"{run_dir}: missing method dir {method_id}/{task_id}")
            continue
        issues.extend(validate_method_dir(method_dir))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate full reproduction outputs.")
    parser.add_argument("--run-dir")
    parser.add_argument("--results-root", default="results_full")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    run_dir = resolve_run_dir(args)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    issues = validate_run(run_dir)
    if issues:
        print(f"Validation failed for {run_dir}")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Validation passed for {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
