from mars_core.api_client import API_CALL_COLUMNS
from mars_core.evaluator import PREDICTION_FIELDS
from mars_core.run_state import REQUIRED_METHOD_FILES


def test_prediction_schema_has_required_fields():
    for field in ["sample_id", "raw_output", "parsed_prediction", "correct"]:
        assert field in PREDICTION_FIELDS


def test_api_call_schema_has_audit_fields():
    for field in ["timestamp", "run_id", "suite", "agent_name", "estimated_cost"]:
        assert field in API_CALL_COLUMNS


def test_required_method_files_include_metrics_and_state():
    assert "metrics.json" in REQUIRED_METHOD_FILES
    assert "run_state.json" in REQUIRED_METHOD_FILES
    assert "api_calls.csv" in REQUIRED_METHOD_FILES
