from mars_core.logging_utils import write_csv, write_json
from mars_core.run_state import method_output_status, should_skip_completed


def test_incomplete_predictions_are_not_complete(tmp_path):
    method_dir = tmp_path / "methods" / "mars_official" / "task"
    method_dir.mkdir(parents=True)
    write_csv(
        method_dir / "predictions.csv",
        [{"sample_id": 1, "error_type": ""}],
        ["sample_id", "error_type"],
    )
    write_json(method_dir / "metrics.json", {"accuracy": 1.0})
    write_json(method_dir / "run_state.json", {"config_hash": "abc"})
    status = method_output_status(method_dir, [1, 2])
    assert status["status"] == "partial"
    assert not should_skip_completed(
        method_dir=method_dir,
        expected_ids=[1, 2],
        config_hash="abc",
        force_rerun=False,
        skip_existing=True,
        resume=False,
    )
