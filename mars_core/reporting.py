from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .logging_utils import write_csv, write_json, write_text


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paper_lookup(paper_results: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for table_name, table_data in (paper_results or {}).items():
        if not isinstance(table_data, dict):
            continue
        for method_id, method_results in table_data.items():
            if not isinstance(method_results, dict):
                continue
            for task_id, accuracy in method_results.items():
                lookup[(method_id, task_id)] = {
                    "paper_table": table_name,
                    "paper_accuracy": accuracy,
                }
    return lookup


def paper_comparison_rows(
    summary_rows: list[dict[str, Any]], paper_results: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = _paper_lookup(paper_results)
    rows = []
    for row in summary_rows:
        suite = str(row.get("suite") or "").strip()
        if suite not in {"main", "ablation"}:
            continue
        method_id = str(row.get("method_id") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        paper = lookup.get((method_id, task_id))
        if not paper:
            continue
        paper_table = str(paper.get("paper_table", ""))
        if suite == "main" and paper_table not in {"table1", "table2"}:
            continue
        if suite == "ablation" and paper_table != "table3":
            continue
        local_accuracy = _as_float(row.get("accuracy"))
        paper_accuracy = _as_float(paper.get("paper_accuracy"))
        if local_accuracy is not None and paper_accuracy is not None:
            paper_fraction = (
                paper_accuracy / 100.0 if paper_accuracy > 1.0 else paper_accuracy
            )
            delta = local_accuracy - paper_fraction
        else:
            delta = None
        rows.append(
            {
                "suite": row.get("suite", ""),
                "paper_table": paper.get("paper_table", ""),
                "task_id": task_id,
                "display_name": row.get("display_name", ""),
                "method_id": method_id,
                "method": row.get("method", method_id),
                "local_accuracy": local_accuracy,
                "paper_accuracy_percent": paper_accuracy,
                "delta_fraction": delta,
                "delta_percentage_points": None if delta is None else delta * 100,
                "model": row.get("model", ""),
                "num_samples": row.get("num_samples", ""),
                "exactness": row.get("exactness", ""),
            }
        )
    return rows


def method_task_coverage(
    summary_rows: list[dict[str, Any]], expected_tasks: list[str], expected_methods: list[str]
) -> dict[str, Any]:
    completed_pairs = {
        (str(row.get("task_id")), str(row.get("method_id")))
        for row in summary_rows
        if row.get("suite") == "main"
        and row.get("task_id")
        and row.get("method_id")
        and not row.get("skipped")
    }
    expected_pairs = {
        (task_id, method_id)
        for task_id in expected_tasks
        for method_id in expected_methods
    }
    missing = sorted(expected_pairs - completed_pairs)
    return {
        "expected_pairs": len(expected_pairs),
        "completed_pairs": len(expected_pairs) - len(missing),
        "coverage_fraction": (len(expected_pairs) - len(missing)) / len(expected_pairs)
        if expected_pairs
        else 1.0,
        "missing_pairs": [
            {"task_id": task_id, "method_id": method_id}
            for task_id, method_id in missing
        ],
    }


def write_full_reproduction_outputs(
    *,
    run_dir: Path,
    args: Any,
    summary_rows: list[dict[str, Any]],
    tasks: dict[str, Any],
    methods: dict[str, Any],
    suites: dict[str, Any],
    paper_results: dict[str, Any],
) -> None:
    comparison = paper_comparison_rows(summary_rows, paper_results)
    write_csv(
        run_dir / "paper_comparison.csv",
        comparison,
        [
            "suite",
            "paper_table",
            "task_id",
            "display_name",
            "method_id",
            "method",
            "local_accuracy",
            "paper_accuracy_percent",
            "delta_fraction",
            "delta_percentage_points",
            "model",
            "num_samples",
            "exactness",
        ],
    )

    suite_counts = Counter(str(row.get("suite", "")) for row in summary_rows)
    exactness_counts = Counter(str(row.get("exactness", "")) for row in summary_rows)
    api_errors = sum(int(row.get("api_errors", 0) or 0) for row in summary_rows)
    parse_errors = sum(int(row.get("parse_errors", 0) or 0) for row in summary_rows)
    main_config = suites.get("main", {})
    main_tasks = (
        list(tasks.keys())
        if main_config.get("tasks") == "all"
        else list(main_config.get("tasks", []))
    )
    main_methods = list(main_config.get("methods", []))
    coverage = method_task_coverage(summary_rows, main_tasks, main_methods)
    write_json(run_dir / "coverage.json", coverage)

    paper_deltas = [
        float(row["delta_percentage_points"])
        for row in comparison
        if row.get("delta_percentage_points") not in (None, "")
        and row.get("method_id") == "mars"
        and row.get("suite") == "main"
    ]
    avg_delta = sum(paper_deltas) / len(paper_deltas) if paper_deltas else None

    lines = [
        "# Full MARS Reproduction Report",
        "",
        "## Run Configuration",
        "",
        f"- suite: {args.suite}",
        f"- model: {args.model}",
        f"- source_model: {getattr(args, 'source_model', None) or args.model}",
        f"- target_models: {getattr(args, 'target_models', '')}",
        f"- temperature: {args.temperature}",
        f"- max_samples: {args.max_samples if args.max_samples is not None else 'all'}",
        f"- max_iterations: {args.max_iterations}",
        f"- eval_protocol: {args.eval_protocol}",
        f"- dry_run: {args.dry_run}",
        f"- cache_enabled: {args.cache_enabled}",
        "",
        "## Matrix Coverage",
        "",
        f"- registered_tasks: {len(tasks)}",
        f"- main_methods: {', '.join(main_methods)}",
        f"- expected_main_task_method_pairs: {coverage['expected_pairs']}",
        f"- completed_main_task_method_pairs: {coverage['completed_pairs']}",
        f"- main_coverage_fraction: {coverage['coverage_fraction']:.4f}",
        "",
        "## Suite Rows",
        "",
    ]
    if suite_counts:
        lines.extend(f"- {suite or 'unknown'}: {count}" for suite, count in suite_counts.items())
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Error Summary",
            "",
            f"- api_errors: {api_errors}",
            f"- parse_errors: {parse_errors}",
            "",
            "## Paper Reference Comparison",
            "",
            f"- comparable_rows: {len(comparison)}",
            f"- mars_average_delta_percentage_points: {avg_delta:.2f}"
            if avg_delta is not None
            else "- mars_average_delta_percentage_points: unavailable",
            "",
            "## Exactness Notes",
            "",
        ]
    )
    if exactness_counts:
        lines.extend(
            f"- {name or 'unspecified'}: {count}"
            for name, count in exactness_counts.items()
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Paper reference numbers are used only for comparison; they are never substituted for failed or missing local runs.",
            "Origin and MARS use local implementations from this repository. APE, ProTeGi, OPRO, PE2, and missing official templates are marked as best-effort reimplementations in `method_config.yaml`.",
        ]
    )
    write_text(run_dir / "paper_reproduction_report.md", "\n".join(lines) + "\n")
    write_text(run_dir / "final_report.md", "\n".join(lines) + "\n")
