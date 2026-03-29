from pydantic import BaseModel, ConfigDict, Field


class TurnContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    session_id: str
    character_id: str
    player_input: str
    max_steps: int = Field(default=4, ge=1)
    debug: bool = False
