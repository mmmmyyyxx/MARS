from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from mars_core.evaluator import compute_final_metrics_from_predictions


def newest_run(results_root: Path) -> Path:
    candidates = sorted(
        (path for path in results_root.glob("run_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No run_* directories found under {results_root}")
    return candidates[0]


def resolve_run_dir(run_dir: str | None, results_root: str, latest: bool) -> Path:
    if run_dir:
        path = Path(run_dir)
        if path.name == "latest":
            return newest_run(path.parent if path.parent != Path(".") else Path(results_root))
        return path
    if latest:
        return newest_run(Path(results_root))
    raise SystemExit("Provide --run-dir or --latest.")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def distribution(values: list[Any], limit: int = 20) -> list[str]:
    counter = Counter("" if value is None else str(value) for value in values)
    if not counter:
        return ["- None"]
    return [
        f"- `{value if value else '<empty>'}`: {count}"
        for value, count in counter.most_common(limit)
    ]


def method_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "methods"
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*/*") if path.is_dir())


def method_report(path: Path) -> list[str]:
    predictions = read_csv_rows(path / "predictions.csv")
    metrics = compute_final_metrics_from_predictions(predictions)
    task_id = path.name
    method_id = path.parent.name
    errors = [row.get("error_type", "") for row in predictions]
    parsed = [row.get("parsed_prediction", "") for row in predictions]
    gold = [row.get("canonical_gold", "") for row in predictions]
    lines = [
        f"## {method_id} / {task_id}",
        "",
        f"- samples: {metrics['num_samples']}",
        f"- accuracy: {metrics['accuracy']}",
        f"- num_correct: {metrics['num_correct']}",
        f"- num_failed: {metrics['num_failed']}",
        f"- api_errors: {metrics['api_errors']}",
        f"- parse_errors: {metrics['parse_errors']}",
        "",
        "### Error Types",
        "",
    ]
    lines.extend(distribution(errors))
    lines.extend(["", "### Parsed Predictions", ""])
    lines.extend(distribution(parsed))
    lines.extend(["", "### Gold Labels", ""])
    lines.extend(distribution(gold))
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose full reproduction prediction parsing and summaries."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--results-root", default="results_full")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    run_dir = resolve_run_dir(args.run_dir, args.results_root, args.latest)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    paths = method_dirs(run_dir)
    lines = [
        "# Full Reproduction Evaluation Diagnostics",
        "",
        f"- run_dir: {run_dir}",
        f"- method_task_dirs: {len(paths)}",
        "",
    ]
    for path in paths:
        lines.extend(method_report(path))

    output_path = run_dir / "evaluation_diagnostics.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation diagnostics written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
