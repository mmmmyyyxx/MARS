from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mars_core.mars_runner import load_task_specs, load_yaml
from mars_core.prompt_loader import PromptLoader
from reproduce_full import preflight_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight full reproduction resources."
    )
    parser.add_argument("--suite", default="all")
    args = parser.parse_args()
    tasks = load_task_specs(ROOT / "configs/tasks.yaml")
    methods = load_yaml(ROOT / "configs/methods.yaml")
    prompt_loader = PromptLoader(ROOT / "Prompt/task_prompts")
    rows = preflight_rows(tasks, methods, prompt_loader)
    columns = [
        "task_id",
        "dataset_exists",
        "origin_prompt_exists",
        "user_prompt_exists",
        "planner_prompt_exists",
        "few_shot_exists",
        "answer_format",
        "methods_supported",
        "runnable",
        "missing_reason",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row[column]) for column in columns))
    return 0 if all(row["runnable"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
