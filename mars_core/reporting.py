from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .api_client import API_CALL_COLUMNS
from .logging_utils import write_csv, write_json, write_text


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


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
        method_id = str(row.get("paper_method_id") or row.get("method_id") or "").strip()
        local_method_id = str(row.get("method_id") or "").strip()
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
                "method_id": local_method_id,
                "paper_method_id": method_id,
                "method": row.get("method", local_method_id),
                "local_accuracy": local_accuracy,
                "paper_accuracy_percent": paper_accuracy,
                "delta_fraction": delta,
                "delta_percentage_points": None if delta is None else delta * 100,
                "model": row.get("model", ""),
                "num_samples": row.get("num_samples", ""),
                "exactness_level": row.get("exactness_level", ""),
                "exactness_note": row.get("exactness_note", ""),
            }
        )
    return rows


def method_task_coverage(
    summary_rows: list[dict[str, Any]], expected_tasks: list[str], expected_methods: list[str]
) -> dict[str, Any]:
    status_by_pair = {}
    for row in summary_rows:
        if row.get("suite") != "main" or not row.get("task_id") or not row.get("method_id"):
            continue
        status_by_pair[(str(row.get("task_id")), str(row.get("method_id")))] = str(
            row.get("status") or ("skipped" if row.get("skipped") else "completed")
        )
    expected_pairs = {
        (task_id, method_id)
        for task_id in expected_tasks
        for method_id in expected_methods
    }
    completed = sorted(
        pair
        for pair in expected_pairs
        if status_by_pair.get(pair) in {"completed", "skipped"}
    )
    partial = sorted(pair for pair in expected_pairs if status_by_pair.get(pair) == "partial")
    failed = sorted(pair for pair in expected_pairs if status_by_pair.get(pair) == "failed")
    skipped = sorted(pair for pair in expected_pairs if status_by_pair.get(pair) == "skipped")
    missing = sorted(pair for pair in expected_pairs if pair not in status_by_pair)
    return {
        "expected_pairs": len(expected_pairs),
        "completed_pairs": len(completed),
        "coverage_fraction": len(completed) / len(expected_pairs) if expected_pairs else 1.0,
        "completed": [
            {"task_id": task_id, "method_id": method_id} for task_id, method_id in completed
        ],
        "partial": [
            {"task_id": task_id, "method_id": method_id} for task_id, method_id in partial
        ],
        "failed": [
            {"task_id": task_id, "method_id": method_id} for task_id, method_id in failed
        ],
        "skipped": [
            {"task_id": task_id, "method_id": method_id} for task_id, method_id in skipped
        ],
        "missing_pairs": [
            {"task_id": task_id, "method_id": method_id} for task_id, method_id in missing
        ],
    }


def _read_api_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def collect_api_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "methods").glob("*/*/api_calls.csv")):
        rows.extend(_read_api_csv(path))
    for path in sorted((run_dir / "transfer").glob("*/methods/*/*/api_calls.csv")):
        rows.extend(_read_api_csv(path))
    return rows


def write_api_summaries(run_dir: Path) -> dict[str, Any]:
    rows = collect_api_rows(run_dir)
    write_csv(run_dir / "api_calls.csv", rows, API_CALL_COLUMNS)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("method_id", "")), str(row.get("model", "")))].append(row)

    token_rows = []
    latency_rows = []
    cost_rows = []
    for (method_id, model), items in sorted(groups.items()):
        prompt_tokens = sum(_as_int(item.get("prompt_tokens")) for item in items)
        completion_tokens = sum(_as_int(item.get("completion_tokens")) for item in items)
        total_tokens = sum(_as_int(item.get("total_tokens")) for item in items)
        estimated_cost = sum(float(item.get("estimated_cost") or 0) for item in items)
        latencies = [
            float(item.get("latency_seconds") or 0)
            for item in items
            if item.get("latency_seconds") not in (None, "")
        ]
        cache_hits = sum(
            str(item.get("cache_hit", "")).strip().lower() in {"true", "1", "yes"}
            for item in items
        )
        token_rows.append(
            {
                "method_id": method_id,
                "model": model,
                "api_call_records": len(items),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        latency_rows.append(
            {
                "method_id": method_id,
                "model": model,
                "api_call_records": len(items),
                "mean_latency_seconds": sum(latencies) / len(latencies)
                if latencies
                else 0.0,
                "max_latency_seconds": max(latencies) if latencies else 0.0,
                "cache_hit_rate": cache_hits / len(items) if items else 0.0,
            }
        )
        cost_rows.append(
            {
                "method_id": method_id,
                "model": model,
                "api_call_records": len(items),
                "estimated_cost": estimated_cost,
            }
        )

    write_csv(
        run_dir / "token_summary.csv",
        token_rows,
        [
            "method_id",
            "model",
            "api_call_records",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ],
    )
    write_csv(
        run_dir / "latency_summary.csv",
        latency_rows,
        [
            "method_id",
            "model",
            "api_call_records",
            "mean_latency_seconds",
            "max_latency_seconds",
            "cache_hit_rate",
        ],
    )
    write_csv(
        run_dir / "cost_summary.csv",
        cost_rows,
        ["method_id", "model", "api_call_records", "estimated_cost"],
    )

    total_rows = len(rows)
    total_tokens = sum(_as_int(row.get("total_tokens")) for row in rows)
    estimated_tokens = sum(_as_int(row.get("estimated_prompt_tokens")) for row in rows) + sum(
        _as_int(row.get("estimated_completion_tokens")) for row in rows
    )
    cache_hits = sum(
        str(row.get("cache_hit", "")).strip().lower() in {"true", "1", "yes"}
        for row in rows
    )
    latencies = [
        float(row.get("latency_seconds") or 0)
        for row in rows
        if row.get("latency_seconds") not in (None, "")
    ]
    total_cost = sum(float(row.get("estimated_cost") or 0) for row in rows)
    return {
        "total_api_call_records": total_rows,
        "total_tokens": total_tokens,
        "total_estimated_tokens": estimated_tokens,
        "cache_hit_rate": cache_hits / total_rows if total_rows else 0.0,
        "total_estimated_cost": total_cost,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def _paper_mode_warning(summary_rows: list[dict[str, Any]], eval_protocol: str) -> str:
    if eval_protocol != "paper_mode":
        return ""
    for row in summary_rows:
        if row.get("opt_hash") and row.get("opt_hash") == row.get("val_hash") == row.get("test_hash"):
            return (
                "Warning: paper_mode is active and at least one row has identical "
                "opt/val/test split hashes. This matches paper-style comparability "
                "but is not leakage-free prompt-search evaluation."
            )
    return ""


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
            "paper_method_id",
            "method",
            "local_accuracy",
            "paper_accuracy_percent",
            "delta_fraction",
            "delta_percentage_points",
            "model",
            "num_samples",
            "exactness_level",
            "exactness_note",
        ],
    )

    api_summary = write_api_summaries(run_dir)
    suite_counts = Counter(str(row.get("suite", "")) for row in summary_rows)
    exactness_counts = Counter(str(row.get("exactness_level", "")) for row in summary_rows)
    status_counts = Counter(str(row.get("status", "")) for row in summary_rows)
    api_errors = sum(_as_int(row.get("api_errors")) for row in summary_rows)
    parse_errors = sum(_as_int(row.get("parse_errors")) for row in summary_rows)
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
        and row.get("paper_method_id") == "mars"
        and row.get("suite") == "main"
    ]
    avg_delta = sum(paper_deltas) / len(paper_deltas) if paper_deltas else None
    warning = _paper_mode_warning(summary_rows, args.eval_protocol)

    method_exactness = []
    for method_id, config in methods.items():
        method_exactness.append(
            {
                "method_id": method_id,
                "display_name": config.get("display_name", method_id),
                "exactness_level": config.get("exactness_level", ""),
                "notes": config.get("exactness_note", ""),
            }
        )
    write_csv(
        run_dir / "exactness_status.csv",
        method_exactness,
        ["method_id", "display_name", "exactness_level", "notes"],
    )

    lines = [
        "# Full MARS Reproduction Report",
        "",
        "This repository supports full local paper-matrix reproduction.",
        "Exact numerical reproduction is only claimed for methods marked exact_official or faithful_reimplementation.",
        "APE, ProTeGi, OPRO, PE2 are currently best-effort unless their original algorithms are implemented.",
        "Paper reference values are used only for comparison and are never substituted for local runs.",
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
        f"- partial_pairs: {len(coverage['partial'])}",
        f"- failed_pairs: {len(coverage['failed'])}",
        f"- skipped_pairs: {len(coverage['skipped'])}",
        "",
        "## Suite Rows",
        "",
    ]
    if suite_counts:
        lines.extend(f"- {suite or 'unknown'}: {count}" for suite, count in suite_counts.items())
    else:
        lines.append("- none")

    lines.extend(["", "## Status", ""])
    if status_counts:
        lines.extend(f"- {status or 'unknown'}: {count}" for status, count in status_counts.items())
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
            "## API Usage",
            "",
            f"- total_api_call_records: {api_summary['total_api_call_records']}",
            f"- total_tokens: {api_summary['total_tokens']}",
            f"- total_estimated_tokens: {api_summary['total_estimated_tokens']}",
            f"- cache_hit_rate: {api_summary['cache_hit_rate']:.4f}",
            f"- total_estimated_cost: {api_summary['total_estimated_cost']:.6f}",
            f"- mean_latency_seconds: {api_summary['mean_latency_seconds']:.4f}",
            "",
            "## Paper Reference Comparison",
            "",
            f"- comparable_rows: {len(comparison)}",
            f"- mars_average_delta_percentage_points: {avg_delta:.2f}"
            if avg_delta is not None
            else "- mars_average_delta_percentage_points: unavailable",
            "",
            "## Exactness",
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
    lines.extend(["", "|Method|Exactness|Notes|", "|---|---|---|"])
    for item in method_exactness:
        lines.append(
            f"|{item['display_name']}|{item['exactness_level']}|{item['notes']}|"
        )

    lines.extend(["", "## Evaluation Protocol", "", f"- protocol: {args.eval_protocol}"])
    if warning:
        lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `summary.csv`, `summary.json`, `paper_comparison.csv`, `coverage.json`",
            "- `api_calls.csv`, `token_summary.csv`, `latency_summary.csv`, `cost_summary.csv`",
            "- method directories under `methods/<method>/<task>/`",
        ]
    )
    write_text(run_dir / "paper_reproduction_report.md", "\n".join(lines) + "\n")
    write_text(run_dir / "final_report.md", "\n".join(lines) + "\n")
