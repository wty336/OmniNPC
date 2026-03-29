from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    TURN_STARTED = "turn_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    ACTION_SELECTED = "action_selected"
    MODEL_CALLED = "model_called"
    TURN_FINISHED = "turn_finished"


class TurnEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    turn_id: str
    step_index: int = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
