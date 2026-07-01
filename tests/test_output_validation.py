import json

from mars_core.logging_utils import write_csv, write_json
from validate_reproduction_outputs import validate_method_dir, ValidationResult


def _complete_method_dir(path):
    path.mkdir(parents=True)
    for filename in [
        "method_config.yaml",
        "output_manifest.json",
        "prompt_accuracy_history.csv",
        "best_prompt.txt",
        "final_prompt.txt",
        "diagnostics.md",
        "api_calls.csv",
        "raw_logs.txt",
    ]:
        (path / filename).write_text("x", encoding="utf-8")
    rows = [
        {
            "sample_id": 0,
            "question": "q",
            "gold": "true",
            "raw_output": "Final answer: true",
            "parsed_prediction": "true",
            "canonical_gold": "true",
            "correct": True,
            "answer_format": "boolean",
            "error_type": "",
            "method": "mars_official",
            "task_id": "boolean_expressions",
            "iteration": 1,
        }
    ]
    write_csv(path / "predictions.csv", rows)
    write_json(
        path / "metrics.json",
        {
            "task_id": "boolean_expressions",
            "method_id": "mars_official",
            "accuracy": 1.0,
            "num_samples": 1,
            "num_correct": 1,
            "num_failed": 0,
            "api_errors": 0,
            "parse_errors": 0,
        },
    )
    write_json(
        path / "run_state.json",
        {
            "run_id": "run",
            "suite": "main",
            "method_id": "mars_official",
            "task_id": "boolean_expressions",
            "model": "dry",
            "temperature": 0,
            "max_samples": 1,
            "dataset_path": "",
            "dataset_hash": "",
            "split_hashes": {},
            "prompt_hash": "",
            "method_config_hash": "",
            "status": "completed",
            "expected_sample_ids": [0],
            "completed_sample_ids": [0],
            "failed_sample_ids": [],
            "started_at": None,
            "completed_at": None,
            "error_message": None,
        },
    )


def test_validator_passes_complete_method_dir(tmp_path):
    method_dir = tmp_path / "method"
    _complete_method_dir(method_dir)
    result = ValidationResult(run_dir=str(tmp_path))
    validate_method_dir(method_dir, "boolean", result)
    assert not result.errors


def test_validator_fails_missing_metrics(tmp_path):
    method_dir = tmp_path / "method"
    _complete_method_dir(method_dir)
    (method_dir / "metrics.json").unlink()
    result = ValidationResult(run_dir=str(tmp_path))
    validate_method_dir(method_dir, "boolean", result)
    assert any("metrics.json" in error for error in result.errors)


def test_validator_fails_duplicate_sample_ids(tmp_path):
    method_dir = tmp_path / "method"
    _complete_method_dir(method_dir)
    rows = list(__import__("csv").DictReader((method_dir / "predictions.csv").open(encoding="utf-8")))
    rows.append(dict(rows[0]))
    write_csv(method_dir / "predictions.csv", rows)
    result = ValidationResult(run_dir=str(tmp_path))
    validate_method_dir(method_dir, "boolean", result)
    assert any("duplicate" in error for error in result.errors)


def test_validator_fails_when_run_scope_expects_more_samples(tmp_path):
    method_dir = tmp_path / "method"
    _complete_method_dir(method_dir)
    result = ValidationResult(run_dir=str(tmp_path))
    validate_method_dir(
        method_dir,
        "boolean",
        result,
        expected_scope={
            "max_samples": None,
            "expected_sample_ids": [0, 1],
            "dataset_hash": "",
            "split_hashes": {},
        },
    )
    assert any("run_config scope" in error for error in result.errors)
