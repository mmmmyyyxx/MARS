import csv
from pathlib import Path

import pytest

from mars_core.mars_runner import (
    TaskSpec,
    evaluate_prompt,
    load_dataset,
    run_direct_method,
    split_dataset,
    split_info,
)
from mars_core.official_mars_runner import parse_steps_strict, run_official_mars
from mars_core.prompt_loader import TaskPrompts


def _task(answer_format="option_letter"):
    return TaskSpec(
        task_id="demo",
        group="BBH",
        paper_table="table1",
        dataset_path="",
        test_path="",
        question_type="choice",
        answer_format=answer_format,
        metric="accuracy",
        user_prompt_key="demo",
        planner_prompt_key="demo",
        few_shot_key="demo",
        paper_display_name="Demo",
    )


def _prompts():
    return TaskPrompts(
        task_id="demo",
        origin="MANUAL_ORIGIN",
        user_proxy="Answer the demo task.",
        planner="Plan for {task_description}",
        cot_zero_shot="Think step by step.",
        cot_few_shot="Use examples.",
        answer_instruction="",
        few_shot_examples=[],
    )


class SequencedClient:
    concurrency = 1
    model = "fake"
    temperature = 0.0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.stats = type("Stats", (), {"call_records": []})()

    def complete_text(self, **kwargs):
        self.calls.append(kwargs)
        if self.outputs:
            return self.outputs.pop(0)
        return "Final answer: (A)"


def test_legacy_initial_prompt_generated_by_student_is_used_by_origin(tmp_path):
    client = SequencedClient(["Final answer: (A)"])
    rows = [{"sample_id": 0, "question": "q (A) yes (B) no", "answer": "(A)"}]

    metrics = run_direct_method(
        client=client,
        task=_task(),
        prompts=_prompts(),
        test_rows=rows,
        method="origin",
        method_config={"display_name": "Origin"},
        out_dir=tmp_path,
        initial_prompt="GENERATED_P0",
        max_answer_retries=1,
    )

    assert metrics["accuracy"] == 1.0
    assert (tmp_path / "best_prompt.txt").read_text(encoding="utf-8") == "GENERATED_P0"
    history = list(csv.DictReader((tmp_path / "prompt_accuracy_history.csv").open()))
    assert history[0]["iteration"] == "0"


def test_mars_iteration_zero_in_history_and_best_prompt_can_remain_p0(tmp_path):
    client = SequencedClient(
        [
            "Total steps: 1\nStep 1: improve",
            "Final answer: (A)",
            "teacher question",
            "ACCEPT",
            "WORSE_PROMPT",
            "Final answer: (B)",
            "Final answer: (A)",
        ]
    )
    rows = [{"sample_id": 0, "question": "q (A) yes (B) no", "answer": "(A)"}]

    run_official_mars(
        client=client,
        task=_task(),
        prompts=_prompts(),
        opt_rows=rows,
        val_rows=rows,
        test_rows=rows,
        method="mars_official",
        method_config={
            "display_name": "MARS",
            "planner_enabled": True,
            "socratic_enabled": True,
            "critic_enabled": True,
        },
        out_dir=tmp_path,
        max_iterations=1,
        max_critic_revisions=1,
        initial_prompt="GENERATED_P0",
        max_answer_retries=1,
        planner_strict_mode=True,
    )

    history = list(csv.DictReader((tmp_path / "prompt_accuracy_history.csv").open()))
    assert history[0]["iteration"] == "0"
    assert history[0]["prompt"] == "GENERATED_P0"
    assert (tmp_path / "best_prompt.txt").read_text(encoding="utf-8") == "GENERATED_P0"


def test_paper_mode_shared_rows():
    rows = [{"sample_id": i, "question": f"q{i}", "answer": "(A)"} for i in range(3)]
    splits = split_dataset(rows, "paper_mode", 42)
    info = split_info(splits, "paper_mode", 42)
    assert info["splits"]["opt"]["hash"] == info["splits"]["val"]["hash"]
    assert info["splits"]["val"]["hash"] == info["splits"]["test"]["hash"]


def test_legacy_skip_first_row(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "question,answer\nq0,(A)\nq1,(A)\nq2,(A)\n",
        encoding="utf-8",
    )

    rows = load_dataset(dataset, skip_first_data_row=True)

    assert len(rows) == 2
    assert [row["question"] for row in rows] == ["q1", "q2"]
    assert [row["sample_id"] for row in rows] == [0, 1]


def test_answer_retry_choice_format_requires_parenthesized_option():
    client = SequencedClient(["A", "Final answer: (A)"])
    rows = [{"sample_id": 0, "question": "q (A) yes (B) no", "answer": "(A)"}]

    predictions = evaluate_prompt(
        client=client,
        task=_task(),
        rows=rows,
        prompt="Choose.",
        method="origin",
        iteration=0,
        max_answer_retries=5,
    )

    assert len(client.calls) == 2
    assert predictions[0]["parsed_prediction"] == "(A)"
    assert predictions[0]["error_type"] == ""
    assert predictions[0]["correct"] is True


def test_planner_strict_format_rejects_missing_total_steps():
    with pytest.raises(ValueError, match="Total steps"):
        parse_steps_strict("Step 1: improve")
