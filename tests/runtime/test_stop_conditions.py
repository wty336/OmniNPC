from src.runtime.step_budget import StepBudget
from src.runtime.stop_conditions import should_stop


def test_should_stop_returns_response_generated_for_respond_actions():
    budget = StepBudget(max_steps=3, used_steps=1)

    assert should_stop("respond", budget) == "response_generated"


def test_should_stop_returns_budget_exhausted_when_no_steps_remain():
    budget = StepBudget(max_steps=2, used_steps=2)

    assert should_stop("retrieve_memory", budget) == "step_budget_exhausted"
