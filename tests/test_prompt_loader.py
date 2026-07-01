from mars_core.mars_runner import load_task_specs
from mars_core.prompt_loader import PromptLoader


def test_prompt_loader_loads_all_registered_tasks():
    loader = PromptLoader()
    for task_id in load_task_specs():
        status = loader.status(task_id)
        assert all(status.values()), task_id
        prompts = loader.load(task_id)
        assert prompts.origin
        assert prompts.user_proxy
        assert prompts.planner
        assert prompts.cot_zero_shot
        assert prompts.cot_few_shot
        assert prompts.answer_instruction
        assert prompts.few_shot_examples
