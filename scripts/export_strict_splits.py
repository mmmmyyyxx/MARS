from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars_core.logging_utils import write_csv, write_json
from mars_core.mars_runner import (
    hash_rows,
    load_dataset,
    load_task_specs,
    split_dataset,
    split_info,
)


def _selected_ids(value: str | None) -> list[str] | None:
    if not value or value.strip().lower() == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_fields(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or [])


def _split_fields(dataset_path: Path) -> list[str]:
    fields = ["sample_id"]
    for field in _dataset_fields(dataset_path):
        if field not in fields:
            fields.append(field)
    return fields


def _summary_row(task_id: str, split_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "split": split_name,
        "num_samples": len(rows),
        "hash": hash_rows(rows),
        "sample_ids": ",".join(str(row["sample_id"]) for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the exact strict_mode opt/val/test CSV splits used by "
            "reproduce_paper.py."
        )
    )
    parser.add_argument("--tasks-config", default="configs/tasks.yaml")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task ids. Defaults to all tasks in configs/tasks.yaml.",
    )
    parser.add_argument("--output-dir", default="strict_splits")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--legacy-skip-first-data-row",
        action="store_true",
        help=(
            "Match paper_mode's legacy data skip if you explicitly used it. "
            "Default strict_mode behavior is not to skip the first data row."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_specs = load_task_specs(ROOT / args.tasks_config)
    selected = _selected_ids(args.tasks)
    if selected is not None:
        missing = [task_id for task_id in selected if task_id not in task_specs]
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(missing)}")
        task_ids = selected
    else:
        task_ids = list(task_specs)

    output_dir = ROOT / args.output_dir
    summary_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "protocol": "strict_mode",
        "split_seed": args.split_seed,
        "max_samples": args.max_samples,
        "legacy_skip_first_data_row": args.legacy_skip_first_data_row,
        "split_rule": "shuffle with random.Random(seed), then 30% opt, 20% val, 50% test",
        "tasks": {},
    }

    for task_id in task_ids:
        task = task_specs[task_id]
        dataset_path = ROOT / task.dataset_path
        rows = load_dataset(
            dataset_path,
            max_samples=args.max_samples,
            skip_first_data_row=args.legacy_skip_first_data_row,
        )
        splits = split_dataset(rows, "strict_mode", args.split_seed)
        task_dir = output_dir / task_id
        fields = _split_fields(dataset_path)
        for split_name, split_rows in splits.items():
            write_csv(task_dir / f"{split_name}.csv", split_rows, fields)
            summary_rows.append(_summary_row(task_id, split_name, split_rows))

        info = split_info(splits, "strict_mode", args.split_seed)
        info["dataset_path"] = task.dataset_path
        info["max_samples"] = args.max_samples
        info["legacy_skip_first_data_row"] = args.legacy_skip_first_data_row
        write_json(task_dir / "split_info.json", info)
        manifest["tasks"][task_id] = info

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        ["task_id", "split", "num_samples", "hash", "sample_ids"],
    )
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote strict splits for {len(task_ids)} task(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
