from mars_core.logging_utils import write_csv, write_json
from mars_core.run_state import (
    build_run_state,
    compatible_with_existing_state,
    missing_sample_ids,
    predictions_complete,
    should_skip_completed,
)


def test_complete_predictions_are_detected(tmp_path):
    predictions = tmp_path / "predictions.csv"
    write_csv(
        predictions,
        [{"sample_id": 1}, {"sample_id": 2}],
        ["sample_id"],
    )
    assert predictions_complete(predictions, [1, 2])
    assert missing_sample_ids(predictions, [1, 2, 3]) == ["3"]


def test_incompatible_state_prevents_skip(tmp_path):
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    write_csv(method_dir / "predictions.csv", [{"sample_id": 1}], ["sample_id"])
    write_json(method_dir / "metrics.json", {"accuracy": 1.0})
    state = build_run_state(
        run_id="run",
        suite="main",
        method_id="mars_official",
        task_id="task",
        model="model-a",
        temperature=0.6,
        max_samples=1,
        dataset_hash="dataset",
        prompt_hash_value="prompt",
        config_hash="config-a",
        expected_ids=[1],
        predictions_path=method_dir / "predictions.csv",
        status="completed",
        split_hashes={"test": "hash"},
    )
    write_json(method_dir / "run_state.json", state.to_dict())
    assert not should_skip_completed(
        method_dir=method_dir,
        expected_ids=[1],
        config_hash="config-b",
        force_rerun=False,
        skip_existing=True,
        resume=False,
    )


def test_compatible_completed_state_allows_skip(tmp_path):
    method_dir = tmp_path / "method"
    method_dir.mkdir()
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
        (method_dir / filename).write_text("x", encoding="utf-8")
    write_csv(method_dir / "predictions.csv", [{"sample_id": 1}], ["sample_id"])
    write_json(method_dir / "metrics.json", {"accuracy": 1.0})
    state = build_run_state(
        run_id="run",
        suite="main",
        method_id="mars_official",
        task_id="task",
        model="model-a",
        temperature=0.6,
        max_samples=1,
        dataset_hash="dataset",
        prompt_hash_value="prompt",
        config_hash="config-a",
        expected_ids=[1],
        predictions_path=method_dir / "predictions.csv",
        status="completed",
    )
    write_json(method_dir / "run_state.json", state.to_dict())
    assert should_skip_completed(
        method_dir=method_dir,
        expected_ids=[1],
        config_hash="config-a",
        force_rerun=False,
        skip_existing=True,
        resume=False,
    )


def test_compatible_with_existing_state_checks_model():
    current = build_run_state(
        run_id="r",
        suite="main",
        method_id="m",
        task_id="t",
        model="a",
        temperature=0.0,
        max_samples=1,
        dataset_hash="d",
        prompt_hash_value="p",
        config_hash="c",
        expected_ids=[],
        predictions_path=__import__("pathlib").Path("missing.csv"),
        status="partial",
    )
    other = build_run_state(
        run_id="r",
        suite="main",
        method_id="m",
        task_id="t",
        model="b",
        temperature=0.0,
        max_samples=1,
        dataset_hash="d",
        prompt_hash_value="p",
        config_hash="c",
        expected_ids=[],
        predictions_path=__import__("pathlib").Path("missing.csv"),
        status="partial",
    )
    assert not compatible_with_existing_state(current, other)
