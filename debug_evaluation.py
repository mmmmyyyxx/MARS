import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from mars_utils.evaluator import (
    build_diagnostics_markdown,
    compute_final_metrics_from_predictions,
    final_prediction_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose MARS prediction parsing and summary consistency.")
    parser.add_argument("--run-dir", help="Specific results_mars/run_* directory, or results_mars/latest.")
    parser.add_argument("--results-root", default="results_mars", help="Root directory containing run_* folders.")
    parser.add_argument("--latest", action="store_true", help="Use the newest run_* directory under --results-root.")
    return parser.parse_args()


def resolve_run_dir(run_dir: Optional[str], results_root: str, latest: bool) -> Path:
    if run_dir:
        path = Path(run_dir)
        if path.name == "latest":
            return newest_run(path.parent)
        return path
    if latest:
        return newest_run(Path(results_root))
    raise SystemExit("Provide --run-dir or --latest.")


def newest_run(results_root: Path) -> Path:
    candidates = sorted(
        (path for path in results_root.glob("run_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No run_* directories found under {results_root}")
    return candidates[0]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def distribution(values: Iterable[str], limit: int = 20) -> List[str]:
    counter = Counter("" if value is None else str(value) for value in values)
    if not counter:
        return ["- None"]
    return [
        f"- `{value if value else '<empty>'}`: {count}"
        for value, count in counter.most_common(limit)
    ]


def summary_by_task(run_dir: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(run_dir / "summary.csv")
    return {row.get("task_id", ""): row for row in rows}


def float_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def task_report(task_dir: Path, summary_row: Dict[str, str]) -> List[str]:
    predictions = read_csv_rows(task_dir / "predictions.csv")
    final_rows = final_prediction_rows(predictions)
    metrics = compute_final_metrics_from_predictions(predictions)
    config = read_yaml(task_dir / "config.yaml")
    answer_formats = sorted({row.get("answer_format", "") for row in final_rows if row.get("answer_format")})
    answer_format = ", ".join(answer_formats) or str(config.get("answer_format") or "unknown")
    parse_failed = sum(
        1
        for row in final_rows
        if not str(row.get("prediction", "")).strip()
        or str(row.get("error", "")).strip() == "answer_parse_failed"
    )

    summary_accuracy = float_or_none(summary_row.get("final_accuracy", ""))
    summary_num_samples = summary_row.get("num_samples", "")
    summary_num_success = summary_row.get("num_success", "")
    consistent = (
        str(metrics["num_samples"]) == str(summary_num_samples)
        and str(metrics["num_success"]) == str(summary_num_success)
        and (
            summary_accuracy is None
            or abs(summary_accuracy - metrics["final_accuracy"]) < 1e-12
        )
    )

    lines = [
        f"## {task_dir.name}",
        "",
        f"- answer_format: {answer_format}",
        f"- final_samples: {metrics['num_samples']}",
        f"- num_success: {metrics['num_success']}",
        f"- num_failed: {metrics['num_failed']}",
        f"- accuracy: {metrics['final_accuracy']}",
        f"- parse_failed: {parse_failed}",
        f"- summary_consistent: {consistent}",
    ]
    if metrics["num_samples"] and metrics["final_accuracy"] == 0:
        lines.append("- warning: accuracy is 0; inspect parsing and raw outputs below.")
    lines.extend(["", "### Gold Label Distribution", ""])
    lines.extend(distribution(row.get("answer", "") for row in final_rows))
    lines.extend(["", "### Canonical Answer Distribution", ""])
    lines.extend(distribution(row.get("canonical_answer", "") for row in final_rows))
    lines.extend(["", "### Canonical Prediction Distribution", ""])
    lines.extend(distribution(row.get("prediction", "") for row in final_rows))
    lines.extend(["", "### Raw Prediction Examples", ""])
    if final_rows:
        for row in final_rows[:20]:
            raw = str(row.get("raw_prediction", "")).replace("\n", "\\n")
            lines.append(f"- `{raw if raw else '<empty>'}`")
    else:
        lines.append("- None")
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir, args.results_root, args.latest)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    summaries = summary_by_task(run_dir)
    task_dirs = sorted((run_dir / "tasks").glob("*"))
    lines = [
        "# MARS Evaluation Diagnostics",
        "",
        f"- run_dir: {run_dir}",
        f"- tasks: {len(task_dirs)}",
        "",
    ]

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        summary_row = summaries.get(task_dir.name, {})
        lines.extend(task_report(task_dir, summary_row))

        predictions = read_csv_rows(task_dir / "predictions.csv")
        (task_dir / "diagnostics.md").write_text(
            build_diagnostics_markdown(task_dir.name, predictions),
            encoding="utf-8",
        )

    output_path = run_dir / "evaluation_diagnostics.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation diagnostics written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
