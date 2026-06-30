from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskPrompts:
    task_id: str
    origin: str
    user_proxy: str
    planner: str
    cot_zero_shot: str
    cot_few_shot: str
    answer_instruction: str
    few_shot_examples: list[dict[str, Any]]


class PromptLoader:
    REQUIRED_TEXT_FILES = [
        "origin.txt",
        "user_proxy.txt",
        "planner.txt",
        "cot_zero_shot.txt",
        "cot_few_shot.txt",
        "answer_instruction.txt",
    ]

    def __init__(self, prompt_root: str | Path = "Prompt/task_prompts"):
        self.prompt_root = Path(prompt_root)

    def task_dir(self, task_id: str) -> Path:
        return self.prompt_root / task_id

    def status(self, task_id: str) -> dict[str, bool]:
        task_dir = self.task_dir(task_id)
        status = {
            name.replace(".txt", "_exists"): (task_dir / name).exists()
            for name in self.REQUIRED_TEXT_FILES
        }
        status["few_shot_exists"] = (task_dir / "few_shot.jsonl").exists()
        return status

    def missing(self, task_id: str) -> list[str]:
        current = self.status(task_id)
        return [name for name, exists in current.items() if not exists]

    def _read_text(self, task_id: str, filename: str) -> str:
        return (self.task_dir(task_id) / filename).read_text(encoding="utf-8").strip()

    def _read_jsonl(self, task_id: str, filename: str) -> list[dict[str, Any]]:
        path = self.task_dir(task_id) / filename
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def load(self, task_id: str) -> TaskPrompts:
        missing = self.missing(task_id)
        if missing:
            raise FileNotFoundError(
                f"Missing prompt resources for {task_id}: {', '.join(missing)}"
            )
        return TaskPrompts(
            task_id=task_id,
            origin=self._read_text(task_id, "origin.txt"),
            user_proxy=self._read_text(task_id, "user_proxy.txt"),
            planner=self._read_text(task_id, "planner.txt"),
            cot_zero_shot=self._read_text(task_id, "cot_zero_shot.txt"),
            cot_few_shot=self._read_text(task_id, "cot_few_shot.txt"),
            answer_instruction=self._read_text(task_id, "answer_instruction.txt"),
            few_shot_examples=self._read_jsonl(task_id, "few_shot.jsonl"),
        )
