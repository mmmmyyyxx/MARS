from pathlib import Path

from mars_core.mars_runner import load_task_specs, load_yaml


def test_all_tasks_have_prompt_directories():
    tasks = load_task_specs("configs/tasks.yaml")
    assert len(tasks) == 17
    for task in tasks.values():
        assert task.dataset_path
        assert task.test_path
        assert task.question_type
        assert task.answer_format
        assert task.user_prompt_key
        assert task.planner_prompt_key
        assert task.few_shot_key
        assert Path(task.dataset_path).exists()


def test_methods_have_exactness_level():
    methods = load_yaml("configs/methods.yaml")
    levels = set(load_yaml("configs/exactness.yaml")["levels"])
    for method_id, config in methods.items():
        assert config["exactness_level"] in levels, method_id


def test_presets_use_official_mars():
    matrix = load_yaml("configs/reproduction_matrix.yaml")
    assert matrix["presets"]["smoke"]["methods"] == ["mars_official"]
    assert "paper_mode" in matrix["protocols"]
