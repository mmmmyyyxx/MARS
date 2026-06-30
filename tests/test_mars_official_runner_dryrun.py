from mars_core.api_client import LLMClient
from mars_core.cache import DiskCache
from mars_core.mars_runner import TaskSpec
from mars_core.official_mars_runner import run_official_mars
from mars_core.prompt_loader import TaskPrompts


def test_official_mars_dryrun_writes_required_files(tmp_path):
    task = TaskSpec(
        task_id="demo",
        group="BBH",
        paper_table="table1",
        dataset_path="",
        test_path="",
        question_type="choice",
        answer_format="option_letter",
        metric="accuracy",
        user_prompt_key="demo",
        planner_prompt_key="demo",
        few_shot_key="demo",
        paper_display_name="Demo",
    )
    prompts = TaskPrompts(
        task_id="demo",
        origin="Choose the best answer.",
        user_proxy="Answer multiple-choice questions.",
        planner="Create prompt-improvement steps.",
        cot_zero_shot="Think step by step.",
        cot_few_shot="Use examples.",
        answer_instruction="",
        few_shot_examples=[],
    )
    rows = [
        {
            "sample_id": 0,
            "question": "Which option is first? (A) one (B) two",
            "answer": "(A)",
        }
    ]
    client = LLMClient(
        model="dry",
        temperature=0,
        cache=DiskCache(tmp_path / ".cache", enabled=False),
        dry_run=True,
        run_id="run_test",
        suite="main",
        method_id="mars_official",
    )
    metrics = run_official_mars(
        client=client,
        task=task,
        prompts=prompts,
        opt_rows=rows,
        val_rows=rows,
        test_rows=rows,
        method="mars_official",
        method_config={
            "display_name": "MARS-official-compatible",
            "exactness_level": "faithful_reimplementation",
            "planner_enabled": True,
            "socratic_enabled": True,
            "critic_enabled": True,
        },
        out_dir=tmp_path / "method",
        max_iterations=2,
        max_critic_revisions=1,
    )
    assert metrics["num_iterations"] == 2
    for filename in [
        "planner_steps.json",
        "teacher_questions.jsonl",
        "critic_feedback.jsonl",
        "student_prompts.jsonl",
        "target_scores.jsonl",
        "best_prompt.txt",
        "final_prompt.txt",
        "predictions.csv",
        "metrics.json",
    ]:
        assert (tmp_path / "method" / filename).exists()
