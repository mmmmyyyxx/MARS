from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .logging_utils import write_json


REQUIRED_METHOD_FILES = [
    "method_config.yaml",
    "run_state.json",
    "metrics.json",
    "predictions.csv",
    "prompt_accuracy_history.csv",
    "best_prompt.txt",
    "final_prompt.txt",
    "diagnostics.md",
    "api_calls.csv",
    "raw_logs.txt",
]


def stable_hash(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def completed_sample_ids(predictions_path: Path) -> list[Any]:
    ids = []
    for row in read_predictions(predictions_path):
        if str(row.get("error_type", "")).strip():
            continue
        ids.append(row.get("sample_id"))
    return ids


def all_prediction_sample_ids(predictions_path: Path) -> list[Any]:
    return [row.get("sample_id") for row in read_predictions(predictions_path)]


def normalize_ids(ids: list[Any]) -> set[str]:
    return {str(item) for item in ids}


def expected_sample_ids(rows: list[dict[str, Any]]) -> list[Any]:
    return [row.get("sample_id") for row in rows]


def build_run_state(
    *,
    run_id: str,
    suite: str,
    method_id: str,
    task_id: str,
    model: str,
    temperature: float,
    max_samples: int | None,
    dataset_hash: str,
    prompt_hash_value: str,
    config_hash: str,
    expected_ids: list[Any],
    predictions_path: Path,
    status: str,
) -> dict[str, Any]:
    present_ids = all_prediction_sample_ids(predictions_path)
    return {
        "run_id": run_id,
        "suite": suite,
        "method_id": method_id,
        "task_id": task_id,
        "model": model,
        "temperature": temperature,
        "max_samples": max_samples,
        "dataset_hash": dataset_hash,
        "prompt_hash": prompt_hash_value,
        "config_hash": config_hash,
        "status": status,
        "completed_sample_ids": present_ids,
        "expected_sample_ids": expected_ids,
    }


def write_run_state(path: Path, state: dict[str, Any]) -> None:
    write_json(path, state)


def load_run_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def method_output_status(method_dir: Path, expected_ids: list[Any]) -> dict[str, Any]:
    missing_files = [
        filename for filename in REQUIRED_METHOD_FILES if not (method_dir / filename).exists()
    ]
    predictions_path = method_dir / "predictions.csv"
    present_ids = all_prediction_sample_ids(predictions_path)
    expected = normalize_ids(expected_ids)
    present = normalize_ids(present_ids)
    missing_ids = sorted(expected - present)
    duplicate_ids = sorted(
        sample_id for sample_id in present if present_ids.count(sample_id) > 1
    )
    if missing_files:
        status = "partial"
    elif missing_ids:
        status = "partial"
    else:
        status = "completed"
    return {
        "status": status,
        "missing_files": missing_files,
        "missing_sample_ids": missing_ids,
        "duplicate_sample_ids": duplicate_ids,
        "num_expected_samples": len(expected_ids),
        "num_prediction_rows": len(present_ids),
    }


def should_skip_completed(
    *,
    method_dir: Path,
    expected_ids: list[Any],
    config_hash: str,
    force_rerun: bool,
    skip_existing: bool,
    resume: bool,
    reuse_compatible_cache: bool = False,
) -> bool:
    if force_rerun or not (skip_existing or resume):
        return False
    metrics_path = method_dir / "metrics.json"
    predictions_path = method_dir / "predictions.csv"
    state = load_run_state(method_dir / "run_state.json")
    if not metrics_path.exists() or not predictions_path.exists() or not state:
        return False
    if (
        not reuse_compatible_cache
        and state.get("config_hash")
        and state.get("config_hash") != config_hash
    ):
        return False
    status = method_output_status(method_dir, expected_ids)
    return status["status"] == "completed"
