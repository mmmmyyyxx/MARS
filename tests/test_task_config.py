from mars_core.mars_runner import load_task_specs, load_yaml


def test_all_tasks_have_prompt_directories():
    tasks = load_task_specs("configs/tasks.yaml")
    assert len(tasks) == 17
    for task in tasks.values():
        assert task.dataset_path
        assert task.answer_format


def test_methods_have_exactness_level():
    methods = load_yaml("configs/methods.yaml")
    levels = set(load_yaml("configs/exactness.yaml")["levels"])
    for method_id, config in methods.items():
        assert config["exactness_level"] in levels, method_id


def test_presets_use_official_mars():
    matrix = load_yaml("configs/reproduction_matrix.yaml")
    assert matrix["presets"]["smoke"]["methods"] == ["mars_official"]
    assert "paper_mode" in matrix["protocols"]
