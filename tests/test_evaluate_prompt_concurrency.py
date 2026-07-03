import threading
import time

from mars_core.mars_runner import TaskSpec, evaluate_prompt


class RecordingClient:
    def __init__(self, concurrency):
        self.concurrency = concurrency
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def complete_text(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return "Final answer: true"
        finally:
            with self.lock:
                self.active -= 1


def test_evaluate_prompt_parallelizes_target_rows_and_preserves_order():
    task = TaskSpec(
        task_id="boolean_expressions",
        group="BBH",
        paper_table="table1",
        dataset_path="",
        test_path="",
        question_type="choice",
        answer_format="boolean",
        metric="accuracy",
        user_prompt_key="boolean_expressions",
        planner_prompt_key="boolean_expressions",
        few_shot_key="boolean_expressions",
        paper_display_name="Boolean Expressions",
    )
    rows = [
        {"sample_id": sample_id, "question": f"q{sample_id}", "answer": "true"}
        for sample_id in range(8)
    ]
    client = RecordingClient(concurrency=4)

    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=rows,
        prompt="Solve.",
        method="mars_official",
        iteration=1,
    )

    assert client.max_active > 1
    assert [row["sample_id"] for row in predictions] == list(range(8))
    assert all(row["correct"] for row in predictions)
