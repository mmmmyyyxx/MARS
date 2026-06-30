from mars_core.prompt_loader import PromptLoader


def test_prompt_loader_loads_registered_task():
    prompts = PromptLoader().load("boolean_expressions")
    assert prompts.origin
    assert prompts.user_proxy
    assert prompts.planner
    assert isinstance(prompts.few_shot_examples, list)
