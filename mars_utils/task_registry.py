from dataclasses import dataclass
import os
from typing import Callable, Iterable, List, Optional

from .config_loader import load_yaml


@dataclass
class MarsTask:
    task_id: str
    group: str
    dataset_path: str
    question_type: str
    user_prompt_key: str
    planner_prompt_key: str
    metric: str = "accuracy"


def load_tasks(path: str) -> List[MarsTask]:
    data = load_yaml(path)
    return [MarsTask(**item) for item in data.get("tasks", [])]


def resolve_tasks(
    selector: str,
    tasks: Iterable[MarsTask],
    is_runnable: Optional[Callable[[MarsTask], bool]] = None,
) -> List[MarsTask]:
    all_tasks = list(tasks)
    requested = [part.strip() for part in selector.split(",") if part.strip()]
    if not requested or requested == ["all"]:
        return all_tasks

    selected: List[MarsTask] = []
    for item in requested:
        lowered = item.lower()
        if lowered == "all":
            selected.extend(all_tasks)
        elif lowered == "runnable":
            selected.extend(task for task in all_tasks if is_runnable and is_runnable(task))
        elif lowered == "bbh":
            selected.extend(task for task in all_tasks if task.group.upper() == "BBH")
        elif lowered == "mmlu":
            selected.extend(task for task in all_tasks if task.group.upper() == "MMLU")
        elif lowered == "domain":
            selected.extend(task for task in all_tasks if task.group.upper() in {"C-EVAL", "GSM8K", "AGIEVAL"})
        else:
            selected.extend(task for task in all_tasks if task.task_id == lowered)

    seen = set()
    unique = []
    for task in selected:
        if task.task_id not in seen:
            unique.append(task)
            seen.add(task.task_id)
    return unique


def dataset_exists(task: MarsTask, root_dir: str) -> bool:
    return os.path.exists(os.path.join(root_dir, task.dataset_path))
