from src.runtime.step_budget import StepBudget


def should_stop(action_type: str, budget: StepBudget) -> str | None:
    if action_type == "respond":
        return "response_generated"
    if budget.remaining_steps == 0:
        return "step_budget_exhausted"
    return None
