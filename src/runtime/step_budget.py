from pydantic import BaseModel, ConfigDict, Field, model_validator


class StepBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=4, ge=1)
    used_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_budget_invariant(self) -> "StepBudget":
        if self.used_steps > self.max_steps:
            raise ValueError("used_steps cannot exceed max_steps")
        return self

    @property
    def remaining_steps(self) -> int:
        return max(self.max_steps - self.used_steps, 0)

    def consume(self) -> None:
        if self.used_steps >= self.max_steps:
            raise ValueError("Step budget exhausted")
        self.used_steps += 1
