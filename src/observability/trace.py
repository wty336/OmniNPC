from typing import Optional

import time

from pydantic import BaseModel, ConfigDict, Field

from src.observability.events import TurnEvent
from src.runtime.turn_context import TurnContext


class TurnTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    session_id: str
    character_id: str
    started_at: float
    ended_at: Optional[float] = None
    events: list[TurnEvent] = Field(default_factory=list)

    @classmethod
    def start(cls, context: TurnContext) -> "TurnTrace":
        return cls(
            turn_id=context.turn_id,
            session_id=context.session_id,
            character_id=context.character_id,
            started_at=time.time(),
        )

    def add_event(self, event: TurnEvent) -> None:
        if event.turn_id != self.turn_id:
            raise ValueError("event turn_id does not match trace turn_id")
        self.events.append(event)

    def finish(self) -> None:
        self.ended_at = time.time()
