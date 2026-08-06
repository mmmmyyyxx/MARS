from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars_core.logging_utils import write_csv, write_text


DEFAULT_TASKS = [
    "boolean_expressions",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "ruin_names",
    "sports_understanding",
]


def _latest_run(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = sorted(path for path in root.glob("run_*") if path.is_dir())
    return runs[-1] if runs else None


def _resolve_run(value: str | None, default_root: str) -> Path | None:
    if value and value.lower() != "latest":
        path = ROOT / value
        return path if path.exists() else Path(value)
    return _latest_run(ROOT / default_root)


def _read_summary(run_dir: Path | None) -> list[dict[str, str]]:
    if run_dir is None:
        return []
    path = run_dir / "summary.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (str(row.get("task_id", "")), str(row.get("method_id", ""))): row
        for row in rows
    }


def _accuracy(
    indexed: dict[tuple[str, str], dict[str, str]], task_id: str, method_id: str
) -> float | None:
    row = indexed.get((task_id, method_id))
    if not row:
        return None
    value = row.get("accuracy", "")
    if value == "":
        return None
    return float(value)


def _display_name(
    indexes: list[dict[tuple[str, str], dict[str, str]]], task_id: str
) -> str:
    for indexed in indexes:
        for (row_task_id, _), row in indexed.items():
            if row_task_id == task_id and row.get("display_name"):
                return str(row["display_name"])
    return task_id


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _average(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(column, "")) for column in columns) + "|")
    return "\n".join(lines) + "\n"


def protocol_rows() -> list[dict[str, str]]:
    return [
        {
            "Protocol": "paper_mode CoT-FS",
            "Few-shot source": 'splits["opt"]',
            "Test overlap": "yes",
            "Role": "Paper-aligned main result; CoT-FS is strongest but opt/val/test share rows.",
        },
        {
            "Protocol": "strict_mode CoT-FS",
            "Few-shot source": 'splits["opt"]',
            "Test overlap": "no",
            "Role": "Removes exact opt/test overlap while preserving split-selected demonstrations.",
        },
        {
            "Protocol": "strict_mode CoT-FS-fixed",
            "Few-shot source": "few_shot.jsonl filtered by test sample_ids",
            "Test overlap": "no",
            "Role": "Fixed demonstrations; removes split-specific demonstration selection.",
        },
    ]


def task_rows(
    tasks: list[str],
    paper: dict[tuple[str, str], dict[str, str]],
    strict: dict[tuple[str, str], dict[str, str]],
    fixed: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paper_cot_values: list[float | None] = []
    paper_mars_values: list[float | None] = []
    strict_cot_values: list[float | None] = []
    strict_mars_values: list[float | None] = []
    fixed_values: list[float | None] = []
    for task_id in tasks:
        paper_cot = _accuracy(paper, task_id, "cot_fs")
        paper_mars = _accuracy(paper, task_id, "mars_official")
        strict_cot = _accuracy(strict, task_id, "cot_fs")
        strict_mars = _accuracy(strict, task_id, "mars_official")
        fixed_cot = _accuracy(fixed, task_id, "cot_fs_fixed")
        paper_cot_values.append(paper_cot)
        paper_mars_values.append(paper_mars)
        strict_cot_values.append(strict_cot)
        strict_mars_values.append(strict_mars)
        fixed_values.append(fixed_cot)
        rows.append(
            {
                "Task": _display_name([paper, strict, fixed], task_id),
                "paper CoT-FS": _fmt(paper_cot),
                "paper MARS": _fmt(paper_mars),
                "paper MARS-CoT": _fmt(_delta(paper_mars, paper_cot)),
                "strict CoT-FS": _fmt(strict_cot),
                "strict MARS": _fmt(strict_mars),
                "strict MARS-CoT": _fmt(_delta(strict_mars, strict_cot)),
                "strict CoT-FS-fixed": _fmt(fixed_cot),
                "fixed-strict CoT": _fmt(_delta(fixed_cot, strict_cot)),
                "MARS-fixed": _fmt(_delta(strict_mars, fixed_cot)),
            }
        )
    rows.append(
        {
            "Task": "Average",
            "paper CoT-FS": _fmt(_average(paper_cot_values)),
            "paper MARS": _fmt(_average(paper_mars_values)),
            "paper MARS-CoT": _fmt(
                _delta(_average(paper_mars_values), _average(paper_cot_values))
            ),
            "strict CoT-FS": _fmt(_average(strict_cot_values)),
            "strict MARS": _fmt(_average(strict_mars_values)),
            "strict MARS-CoT": _fmt(
                _delta(_average(strict_mars_values), _average(strict_cot_values))
            ),
            "strict CoT-FS-fixed": _fmt(_average(fixed_values)),
            "fixed-strict CoT": _fmt(
                _delta(_average(fixed_values), _average(strict_cot_values))
            ),
            "MARS-fixed": _fmt(_delta(_average(strict_mars_values), _average(fixed_values))),
        }
    )
    return rows


def interpretation_rows(
    comparison_rows: list[dict[str, str]], meaningful_delta: float
) -> list[dict[str, str]]:
    average = comparison_rows[-1]
    strict_cot = _parse_optional_float(average["strict CoT-FS"])
    fixed_cot = _parse_optional_float(average["strict CoT-FS-fixed"])
    strict_mars = _parse_optional_float(average["strict MARS"])

    if fixed_cot is None:
        status_a = status_b = status_c = "pending"
        evidence = "cot_fs_fixed real results are not available yet."
        return [
            {
                "Case": "A",
                "Condition": "cot_fs_fixed clearly below strict CoT-FS",
                "Status": status_a,
                "Evidence": evidence,
                "Interpretation": "Run cot_fs_fixed to test whether split-selected opt examples drive CoT-FS strength.",
            },
            {
                "Case": "B",
                "Condition": "cot_fs_fixed close to strict CoT-FS",
                "Status": status_b,
                "Evidence": evidence,
                "Interpretation": "Run cot_fs_fixed to test whether few-shot prompting itself is the main driver.",
            },
            {
                "Case": "C",
                "Condition": "cot_fs_fixed below strict MARS",
                "Status": status_c,
                "Evidence": evidence,
                "Interpretation": "Run cot_fs_fixed to test whether MARS beats fixed non-overlapping demonstrations.",
            },
        ]

    fixed_minus_strict = fixed_cot - strict_cot if strict_cot is not None else None
    mars_minus_fixed = strict_mars - fixed_cot if strict_mars is not None else None
    case_a = (
        fixed_minus_strict is not None and fixed_minus_strict <= -meaningful_delta
    )
    case_b = fixed_minus_strict is not None and abs(fixed_minus_strict) < meaningful_delta
    case_c = mars_minus_fixed is not None and mars_minus_fixed > 0
    return [
        {
            "Case": "A",
            "Condition": "cot_fs_fixed clearly below strict CoT-FS",
            "Status": "yes" if case_a else "no",
            "Evidence": f"fixed - strict CoT-FS = {_fmt(fixed_minus_strict)}",
            "Interpretation": "Supports demonstration-source sensitivity." if case_a else "Not supported by the current average.",
        },
        {
            "Case": "B",
            "Condition": "cot_fs_fixed close to strict CoT-FS",
            "Status": "yes" if case_b else "no",
            "Evidence": f"fixed - strict CoT-FS = {_fmt(fixed_minus_strict)}",
            "Interpretation": "Supports intrinsic few-shot strength." if case_b else "Not supported by the current average.",
        },
        {
            "Case": "C",
            "Condition": "cot_fs_fixed below strict MARS",
            "Status": "yes" if case_c else "no",
            "Evidence": f"MARS - fixed = {_fmt(mars_minus_fixed)}",
            "Interpretation": "Strongly supports the MARS reproduction explanation." if case_c else "MARS does not beat fixed CoT-FS on average.",
        },
    ]


def _parse_optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build diagnostic tables for paper/strict/fixed CoT-FS comparisons."
    )
    parser.add_argument("--paper-run", default="latest")
    parser.add_argument("--strict-run", default="latest")
    parser.add_argument("--fixed-run", default="latest")
    parser.add_argument("--paper-root", default="paper_full")
    parser.add_argument("--strict-root", default="paper_full_strict")
    parser.add_argument("--fixed-root", default="paper_full_strict_fixed")
    parser.add_argument("--output-dir", default="fewshot_diagnostics")
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task ids.",
    )
    parser.add_argument(
        "--meaningful-delta",
        type=float,
        default=0.02,
        help="Average accuracy difference threshold for clear vs close.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    paper_run = _resolve_run(args.paper_run, args.paper_root)
    strict_run = _resolve_run(args.strict_run, args.strict_root)
    fixed_run = _resolve_run(args.fixed_run, args.fixed_root)
    paper = _index(_read_summary(paper_run))
    strict = _index(_read_summary(strict_run))
    fixed = _index(_read_summary(fixed_run))

    out_dir = ROOT / args.output_dir
    protocols = protocol_rows()
    comparisons = task_rows(tasks, paper, strict, fixed)
    interpretations = interpretation_rows(comparisons, args.meaningful_delta)

    write_csv(
        out_dir / "table1_protocols.csv",
        protocols,
        ["Protocol", "Few-shot source", "Test overlap", "Role"],
    )
    write_csv(
        out_dir / "table2_task_accuracy.csv",
        comparisons,
        [
            "Task",
            "paper CoT-FS",
            "paper MARS",
            "paper MARS-CoT",
            "strict CoT-FS",
            "strict MARS",
            "strict MARS-CoT",
            "strict CoT-FS-fixed",
            "fixed-strict CoT",
            "MARS-fixed",
        ],
    )
    write_csv(
        out_dir / "table3_interpretation.csv",
        interpretations,
        ["Case", "Condition", "Status", "Evidence", "Interpretation"],
    )

    protocol_md = _markdown_table(
        protocols, ["Protocol", "Few-shot source", "Test overlap", "Role"]
    )
    comparison_md = _markdown_table(
        comparisons,
        [
            "Task",
            "paper CoT-FS",
            "paper MARS",
            "paper MARS-CoT",
            "strict CoT-FS",
            "strict MARS",
            "strict MARS-CoT",
            "strict CoT-FS-fixed",
            "fixed-strict CoT",
            "MARS-fixed",
        ],
    )
    interpretation_md = _markdown_table(
        interpretations, ["Case", "Condition", "Status", "Evidence", "Interpretation"]
    )
    report = (
        "# Few-shot Diagnostic Tables\n\n"
        f"- paper run: `{paper_run or 'missing'}`\n"
        f"- strict run: `{strict_run or 'missing'}`\n"
        f"- fixed run: `{fixed_run or 'missing'}`\n"
        f"- meaningful delta: `{args.meaningful_delta}`\n\n"
        "## Table 1. Protocols\n\n"
        f"{protocol_md}\n"
        "## Table 2. Task Accuracy\n\n"
        f"{comparison_md}\n"
        "## Table 3. Interpretation\n\n"
        f"{interpretation_md}\n"
    )
    write_text(out_dir / "README.md", report)
    print(f"Wrote few-shot diagnostic tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
