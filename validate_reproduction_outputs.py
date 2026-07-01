from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars_core.evaluator import PREDICTION_FIELDS, compute_final_metrics_from_predictions
from mars_core.mars_runner import hash_rows, load_dataset, load_task_specs, load_yaml, split_dataset
from mars_core.run_state import (
    REQUIRED_METHOD_FILES,
    load_run_state,
    predictions_complete,
)


@dataclass
class ValidationResult:
    run_dir: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed,
        }


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


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_id_set(values: list[Any]) -> set[str]:
    return {str(value) for value in values}


def expected_task_scope(
    task_id: str,
    run_config: dict[str, Any],
    tasks: dict[str, Any],
) -> dict[str, Any]:
    task = tasks[task_id]
    max_samples = parse_optional_int(run_config.get("max_samples"))
    eval_protocol = str(run_config.get("eval_protocol") or "paper_mode")
    split_seed = int(run_config.get("split_seed") or 42)
    all_rows = load_dataset(task.dataset_path, max_samples)
    splits = split_dataset(all_rows, eval_protocol, split_seed)
    return {
        "max_samples": max_samples,
        "expected_sample_ids": [row.get("sample_id") for row in splits["test"]],
        "dataset_hash": hash_rows(all_rows),
        "split_hashes": {
            "opt": hash_rows(splits.get("opt", [])),
            "val": hash_rows(splits.get("val", [])),
            "test": hash_rows(splits.get("test", [])),
        },
    }


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


def expected_answer_formats() -> dict[str, str]:
    return {task_id: task.answer_format for task_id, task in load_task_specs().items()}


def validate_predictions(
    *,
    method_dir: Path,
    metrics: dict[str, Any],
    task_answer_format: str,
    result: ValidationResult,
    expected_sample_ids: list[Any] | None = None,
) -> None:
    predictions_path = method_dir / "predictions.csv"
    predictions = read_csv_rows(predictions_path)
    if not predictions:
        result.errors.append(f"{method_dir}: predictions.csv has no rows")
        return
    missing_columns = [field for field in PREDICTION_FIELDS if field not in predictions[0]]
    if missing_columns:
        result.errors.append(f"{method_dir}: predictions.csv missing columns {missing_columns}")
    sample_ids = [row.get("sample_id", "") for row in predictions]
    duplicate_ids = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    if duplicate_ids:
        result.errors.append(f"{method_dir}: duplicate sample_id values {duplicate_ids}")
    if len(set(sample_ids)) != int(metrics.get("num_samples", -1)):
        result.errors.append(
            f"{method_dir}: unique sample_id count does not equal metrics.num_samples"
        )
    if expected_sample_ids is not None:
        expected_ids = normalize_id_set(expected_sample_ids)
        if set(sample_ids) != expected_ids or len(sample_ids) != len(expected_sample_ids):
            result.errors.append(
                f"{method_dir}: predictions sample_id set does not match run_config scope "
                f"(expected {len(expected_sample_ids)}, got {len(sample_ids)})"
            )
        if int(metrics.get("num_samples", -1)) != len(expected_sample_ids):
            result.errors.append(
                f"{method_dir}: metrics.num_samples does not match run_config scope "
                f"(expected {len(expected_sample_ids)}, got {metrics.get('num_samples')})"
            )
    for row in predictions:
        if parse_bool(row.get("correct")) is None:
            result.errors.append(f"{method_dir}: unparseable correct value {row.get('correct')!r}")
            break
        if row.get("answer_format") != task_answer_format:
            result.errors.append(
                f"{method_dir}: answer_format {row.get('answer_format')} != {task_answer_format}"
            )
            break
    recomputed = compute_final_metrics_from_predictions(predictions)
    expected_accuracy = float(metrics.get("accuracy", -1))
    if abs(float(recomputed["accuracy"]) - expected_accuracy) > 1e-9:
        result.errors.append(f"{method_dir}: metrics accuracy does not match predictions")


def validate_state(
    *,
    method_dir: Path,
    metrics: dict[str, Any],
    result: ValidationResult,
    expected_sample_ids: list[Any] | None = None,
    expected_max_samples: int | None = None,
    expected_dataset_hash: str = "",
    expected_split_hashes: dict[str, str] | None = None,
) -> None:
    state = load_run_state(method_dir / "run_state.json")
    if state is None:
        result.errors.append(f"{method_dir}: run_state.json is missing or invalid")
        return
    predictions_path = method_dir / "predictions.csv"
    complete = predictions_complete(predictions_path, state.expected_sample_ids)
    if state.status == "completed" and not complete:
        result.errors.append(f"{method_dir}: run_state completed but predictions incomplete")
    prediction_ids = {
        str(row.get("sample_id")) for row in read_csv_rows(predictions_path)
    }
    completed_ids = {str(item) for item in state.completed_sample_ids}
    if completed_ids != prediction_ids:
        result.errors.append(f"{method_dir}: run_state completed_sample_ids mismatch")
    if state.method_id != metrics.get("method_id"):
        result.errors.append(f"{method_dir}: run_state method_id mismatch")
    if state.task_id != metrics.get("task_id"):
        result.errors.append(f"{method_dir}: run_state task_id mismatch")
    if expected_sample_ids is not None and normalize_id_set(
        state.expected_sample_ids
    ) != normalize_id_set(expected_sample_ids):
        result.errors.append(
            f"{method_dir}: run_state expected_sample_ids do not match run_config scope "
            f"(expected {len(expected_sample_ids)}, got {len(state.expected_sample_ids)})"
        )
    if expected_sample_ids is not None and parse_optional_int(
        state.max_samples
    ) != expected_max_samples:
        result.errors.append(
            f"{method_dir}: run_state max_samples {state.max_samples!r} "
            f"!= run_config max_samples {expected_max_samples!r}"
        )
    if expected_dataset_hash and state.dataset_hash != expected_dataset_hash:
        result.errors.append(f"{method_dir}: run_state dataset_hash mismatch")
    if expected_split_hashes and state.split_hashes != expected_split_hashes:
        result.errors.append(f"{method_dir}: run_state split_hashes mismatch")


def validate_method_dir(
    method_dir: Path,
    task_answer_format: str,
    result: ValidationResult,
    expected_scope: dict[str, Any] | None = None,
) -> None:
    for filename in REQUIRED_METHOD_FILES:
        if not (method_dir / filename).exists():
            result.errors.append(f"{method_dir}: missing {filename}")
    metrics_path = method_dir / "metrics.json"
    if not metrics_path.exists():
        return
    try:
        metrics = read_json(metrics_path)
    except json.JSONDecodeError:
        result.errors.append(f"{method_dir}: metrics.json is invalid JSON")
        return
    validate_predictions(
        method_dir=method_dir,
        metrics=metrics,
        task_answer_format=task_answer_format,
        result=result,
        expected_sample_ids=None
        if expected_scope is None
        else expected_scope["expected_sample_ids"],
    )
    validate_state(
        method_dir=method_dir,
        metrics=metrics,
        result=result,
        expected_sample_ids=None
        if expected_scope is None
        else expected_scope["expected_sample_ids"],
        expected_max_samples=None
        if expected_scope is None
        else expected_scope["max_samples"],
        expected_dataset_hash="" if expected_scope is None else expected_scope["dataset_hash"],
        expected_split_hashes=None
        if expected_scope is None
        else expected_scope["split_hashes"],
    )
    best_prompt = method_dir / "best_prompt.txt"
    if best_prompt.exists() and not best_prompt.read_text(encoding="utf-8").strip():
        result.errors.append(f"{method_dir}: best_prompt.txt is empty")
    api_calls = method_dir / "api_calls.csv"
    if api_calls.exists() and api_calls.stat().st_size == 0:
        result.warnings.append(f"{method_dir}: api_calls.csv is empty")


def validate_run(run_dir: Path) -> ValidationResult:
    result = ValidationResult(run_dir=str(run_dir))
    run_config = load_yaml(run_dir / "run_config.yaml") if (run_dir / "run_config.yaml").exists() else {}
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
            result.errors.append(f"{run_dir}: missing {filename}")
    for dirname in ["tables", "figures", "methods", "tasks"]:
        if not (run_dir / dirname).exists():
            result.errors.append(f"{run_dir}: missing directory {dirname}")

    tasks = load_task_specs()
    answer_formats = {task_id: task.answer_format for task_id, task in tasks.items()}
    expected_scopes = {
        task_id: expected_task_scope(task_id, run_config, tasks) for task_id in tasks
    }
    for task_id, method_id in sorted(expected_main_pairs(run_dir)):
        method_dir = run_dir / "methods" / method_id / task_id
        if not method_dir.exists():
            result.errors.append(f"{run_dir}: missing method dir {method_id}/{task_id}")
            continue
        validate_method_dir(
            method_dir,
            answer_formats.get(task_id, ""),
            result,
            expected_scope=expected_scopes.get(task_id),
        )
    return result


def write_reports(run_dir: Path, result: ValidationResult) -> None:
    (run_dir / "validation_report.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Validation Report",
        "",
        f"- run_dir: {result.run_dir}",
        f"- errors: {len(result.errors)}",
        f"- warnings: {len(result.warnings)}",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {item}" for item in result.errors) if result.errors else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in result.warnings) if result.warnings else lines.append("- none")
    (run_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_lines = [
        "",
        "## Output Validation Summary (Latest Validation)",
        "",
        f"- validation_errors: {len(result.errors)}",
        f"- validation_warnings: {len(result.warnings)}",
        f"- validation_passed: {not result.errors and not result.warnings}",
        "",
    ]
    for report_name in ["final_report.md", "paper_reproduction_report.md"]:
        report_path = run_dir / report_name
        if not report_path.exists():
            continue
        text = report_path.read_text(encoding="utf-8")
        marker = "## Output Validation Summary (Latest Validation)"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip() + "\n"
        report_path.write_text(text.rstrip() + "\n" + "\n".join(summary_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate full reproduction outputs.")
    parser.add_argument("--run-dir")
    parser.add_argument("--results-root", default="results_full")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    run_dir = resolve_run_dir(args)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    result = validate_run(run_dir)
    write_reports(run_dir, result)
    if result.errors:
        print(f"Validation failed for {run_dir}: {len(result.errors)} errors")
        return 2
    if result.warnings:
        print(f"Validation passed with warnings for {run_dir}: {len(result.warnings)} warnings")
        return 1 if args.strict else 0
    print(f"Validation passed for {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
