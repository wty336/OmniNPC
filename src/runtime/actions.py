from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_respond_payload(self) -> "RuntimeAction":
        if self.action_type == "respond":
            dialogue = self.payload.get("dialogue")
            if not isinstance(dialogue, str) or not dialogue.strip():
                raise ValueError('respond actions require a non-empty payload["dialogue"]')
        return self
