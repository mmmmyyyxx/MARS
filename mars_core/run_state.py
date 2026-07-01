from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging_utils import write_json


REQUIRED_METHOD_FILES = [
    "method_config.yaml",
    "run_state.json",
    "metrics.json",
    "output_manifest.json",
    "predictions.csv",
    "prompt_accuracy_history.csv",
    "best_prompt.txt",
    "final_prompt.txt",
    "diagnostics.md",
    "api_calls.csv",
    "raw_logs.txt",
]


@dataclass
class RunState:
    run_id: str
    suite: str
    method_id: str
    task_id: str
    model: str
    temperature: float
    max_samples: int | None
    dataset_path: str
    dataset_hash: str
    split_hashes: dict[str, str]
    prompt_hash: str
    method_config_hash: str
    status: str
    expected_sample_ids: list[Any]
    completed_sample_ids: list[Any]
    failed_sample_ids: list[Any]
    started_at: str | None
    completed_at: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_hash"] = self.method_config_hash
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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


def normalize_ids(ids: list[Any]) -> set[str]:
    return {str(item) for item in ids}


def expected_sample_ids(rows: list[dict[str, Any]]) -> list[Any]:
    return [row.get("sample_id") for row in rows]


def all_prediction_sample_ids(predictions_path: Path) -> list[Any]:
    return [row.get("sample_id") for row in read_predictions(predictions_path)]


def completed_sample_ids(predictions_path: Path) -> list[Any]:
    ids = []
    for row in read_predictions(predictions_path):
        if str(row.get("error_type", "")).strip():
            continue
        ids.append(row.get("sample_id"))
    return ids


def failed_sample_ids(predictions_path: Path) -> list[Any]:
    ids = []
    for row in read_predictions(predictions_path):
        if str(row.get("error_type", "")).strip():
            ids.append(row.get("sample_id"))
    return ids


def missing_sample_ids(predictions_path: Path, expected_ids: list[Any]) -> list[str]:
    present = normalize_ids(all_prediction_sample_ids(predictions_path))
    return sorted(normalize_ids(expected_ids) - present)


def predictions_complete(predictions_path: Path, expected_ids: list[Any]) -> bool:
    if not predictions_path.exists():
        return False
    rows = read_predictions(predictions_path)
    if not rows and expected_ids:
        return False
    present = [str(row.get("sample_id")) for row in rows]
    return (
        normalize_ids(expected_ids) == set(present)
        and len(present) == len(set(present))
    )


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
    dataset_path: str = "",
    split_hashes: dict[str, str] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    error_message: str | None = None,
) -> RunState:
    if status == "completed" and not predictions_complete(predictions_path, expected_ids):
        status = "partial"
    return RunState(
        run_id=run_id,
        suite=suite,
        method_id=method_id,
        task_id=task_id,
        model=model,
        temperature=temperature,
        max_samples=max_samples,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        split_hashes=split_hashes or {},
        prompt_hash=prompt_hash_value,
        method_config_hash=config_hash,
        status=status,
        expected_sample_ids=expected_ids,
        completed_sample_ids=all_prediction_sample_ids(predictions_path),
        failed_sample_ids=failed_sample_ids(predictions_path),
        started_at=started_at,
        completed_at=completed_at or (utc_now() if status == "completed" else None),
        error_message=error_message,
    )


def save_run_state(path: Path, state: RunState | dict[str, Any]) -> None:
    data = state.to_dict() if isinstance(state, RunState) else state
    write_json(path, data)


def write_run_state(path: Path, state: RunState | dict[str, Any]) -> None:
    save_run_state(path, state)


def load_run_state(path: Path) -> RunState | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "method_config_hash" not in data and "config_hash" in data:
        data["method_config_hash"] = data["config_hash"]
    defaults = {
        "dataset_path": "",
        "split_hashes": {},
        "failed_sample_ids": [],
        "started_at": None,
        "completed_at": None,
        "error_message": None,
    }
    for key, value in defaults.items():
        data.setdefault(key, value)
    try:
        return RunState(
            run_id=data.get("run_id", ""),
            suite=data.get("suite", ""),
            method_id=data.get("method_id", ""),
            task_id=data.get("task_id", ""),
            model=data.get("model", ""),
            temperature=float(data.get("temperature", 0) or 0),
            max_samples=data.get("max_samples"),
            dataset_path=data.get("dataset_path", ""),
            dataset_hash=data.get("dataset_hash", ""),
            split_hashes=dict(data.get("split_hashes", {}) or {}),
            prompt_hash=data.get("prompt_hash", ""),
            method_config_hash=data.get("method_config_hash", ""),
            status=data.get("status", ""),
            expected_sample_ids=list(data.get("expected_sample_ids", []) or []),
            completed_sample_ids=list(data.get("completed_sample_ids", []) or []),
            failed_sample_ids=list(data.get("failed_sample_ids", []) or []),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
        )
    except (TypeError, ValueError):
        return None


def compatible_with_existing_state(
    existing: RunState, current: RunState, *, reuse_compatible_cache: bool = False
) -> bool:
    if existing.method_id != current.method_id:
        return False
    if existing.task_id != current.task_id:
        return False
    if existing.model != current.model:
        return False
    if float(existing.temperature) != float(current.temperature):
        return False
    if existing.dataset_hash != current.dataset_hash:
        return False
    if existing.split_hashes != current.split_hashes:
        return False
    if reuse_compatible_cache:
        return True
    return existing.method_config_hash == current.method_config_hash


def method_output_status(method_dir: Path, expected_ids: list[Any]) -> dict[str, Any]:
    missing_files = [
        filename
        for filename in REQUIRED_METHOD_FILES
        if not (method_dir / filename).exists()
    ]
    predictions_path = method_dir / "predictions.csv"
    present_ids = all_prediction_sample_ids(predictions_path)
    expected = normalize_ids(expected_ids)
    present = normalize_ids(present_ids)
    missing_ids = sorted(expected - present)
    duplicate_ids = sorted(
        sample_id for sample_id in present if present_ids.count(sample_id) > 1
    )
    if missing_files or missing_ids or duplicate_ids:
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
        and state.method_config_hash
        and state.method_config_hash != config_hash
    ):
        return False
    return (
        state.status == "completed"
        and predictions_complete(predictions_path, expected_ids)
        and method_output_status(method_dir, expected_ids)["status"] == "completed"
    )
