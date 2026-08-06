import csv
from pathlib import Path

import pytest

from mars_core.mars_runner import (
    build_legacy_target_prompt,
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


def _prompts_with_few_shot_examples():
    prompts = _prompts()
    prompts.few_shot_examples.extend(
        [
            {"sample_id": 0, "question": "test overlap", "answer": "(A)"},
            {"sample_id": 1, "question": "external one", "answer": "(A)"},
            {"sample_id": 2, "question": "external two", "answer": "(A)"},
            {"sample_id": 3, "question": "external three", "answer": "(A)"},
        ]
    )
    return prompts


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


def test_cot_fs_fixed_filters_current_test_sample_ids(tmp_path):
    client = SequencedClient(["Final answer: (A)"])
    rows = [{"sample_id": 0, "question": "q (A) yes (B) no", "answer": "(A)"}]

    run_direct_method(
        client=client,
        task=_task(),
        prompts=_prompts_with_few_shot_examples(),
        test_rows=rows,
        method="cot_fs_fixed",
        method_config={"display_name": "CoT(FS-fixed)", "num_shots": 3},
        out_dir=tmp_path,
        max_answer_retries=1,
    )

    used_examples = [
        line for line in (tmp_path / "used_few_shot_examples.jsonl").read_text().splitlines()
        if line
    ]
    assert len(used_examples) == 3
    assert "test overlap" not in (tmp_path / "best_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "external three" in (tmp_path / "best_prompt.txt").read_text(
        encoding="utf-8"
    )


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


def test_task_prompts_use_legacy_userproxy_and_planner_blocks():
    user_proxy = Path(
        "Prompt/task_prompts/geometric_shapes/user_proxy.txt"
    ).read_text(encoding="utf-8")
    planner = Path("Prompt/task_prompts/geometric_shapes/planner.txt").read_text(
        encoding="utf-8"
    )

    assert "This SVG path element" in user_proxy
    assert "Options:" in user_proxy
    assert "Total steps: 5" in planner
    assert "SVG path element" in planner


def test_legacy_target_prompt_uses_original_system_and_concat():
    client = SequencedClient(["(A)"])
    rows = [{"sample_id": 0, "question": "q (A) yes (B) no", "answer": "(A)"}]

    predictions = evaluate_prompt(
        client=client,
        task=_task(),
        rows=rows,
        prompt="PROMPT",
        method="origin",
        iteration=0,
        max_answer_retries=5,
        legacy_target_prompt_mode=True,
    )

    assert predictions[0]["correct"] is True
    call = client.calls[0]
    assert call["system"] == "You are a helpful assistant"
    assert call["user"].startswith("PROMPT\nQuestion: q")
    assert "parenthesis containing the capital letter" in call["user"]


def test_legacy_target_prompt_keeps_non_option_tasks_short_answer_style():
    task = _task(answer_format="boolean")
    prompt = build_legacy_target_prompt(task, "PROMPT", "True and False is")

    assert "parenthesis containing the capital letter" not in prompt
    assert "only the content of the answer" in prompt


def test_mars_uses_legacy_agent_prompts_and_target_call_limit(tmp_path):
    client = SequencedClient(
        [
            "Total steps: 2\nStep 1: improve one\nStep 2: improve two",
            "(A)",
            "Q1?\nQ2?",
            "[True]",
            "PROMPT_1",
            "(B)",
            "(A)",
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
        max_iterations=10,
        max_critic_revisions=1,
        initial_prompt="GENERATED_P0",
        max_answer_retries=1,
        planner_strict_mode=True,
        legacy_target_prompt_mode=True,
        target_call_count_limit=2,
    )

    history = list(csv.DictReader((tmp_path / "prompt_accuracy_history.csv").open()))
    assert [row["iteration"] for row in history] == ["0", "1"]
    raw_logs = (tmp_path / "raw_logs.txt").read_text(encoding="utf-8")
    assert "target_call_count: 2" in raw_logs

    teacher_call = next(call for call in client.calls if call["agent_name"] == "Teacher")
    critic_call = next(call for call in client.calls if call["agent_name"] == "Critic")
    student_call = [
        call for call in client.calls if call["agent_name"] == "Student"
    ][0]
    target_call = next(call for call in client.calls if call["agent_name"] == "Target")

    assert "Please ask a total of two questions" in teacher_call["system"]
    assert "Ask heuristic questions" in teacher_call["user"]
    assert "[True]" in critic_call["system"]
    assert "questions:" in critic_call["user"]
    assert "prompt generator" in student_call["system"]
    assert "Please base on the following question update your prompt" in student_call["user"]
    assert target_call["system"] == "You are a helpful assistant"


def test_critic_false_regenerates_teacher_once_without_second_critic(tmp_path):
    client = SequencedClient(
        [
            "Total steps: 1\nStep 1: improve",
            "(A)",
            "BAD QUESTION",
            "[False]\n[suggestion: ask two Socratic questions]",
            "BETTER QUESTION",
            "PROMPT_1",
            "(A)",
            "(A)",
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
        legacy_target_prompt_mode=True,
    )

    teacher_calls = [call for call in client.calls if call["agent_name"] == "Teacher"]
    critic_calls = [call for call in client.calls if call["agent_name"] == "Critic"]
    student_calls = [call for call in client.calls if call["agent_name"] == "Student"]

    assert len(teacher_calls) == 2
    assert len(critic_calls) == 1
    assert "feedback on whether your output matches" in teacher_calls[1]["user"]
    assert "BETTER QUESTION" in student_calls[-1]["user"]
