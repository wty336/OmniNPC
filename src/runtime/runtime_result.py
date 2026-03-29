from pydantic import BaseModel, ConfigDict

from src.observability.trace import TurnTrace


class RuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dialogue: str
    trace: TurnTrace
    stop_reason: str
