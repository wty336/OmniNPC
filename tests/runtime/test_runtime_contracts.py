from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.observability.events import EventType, TurnEvent
from src.observability.trace import TurnTrace
from src.runtime.runtime_result import RuntimeResult
from src.runtime.step_budget import StepBudget
from src.runtime.turn_context import TurnContext


def test_turn_context_and_trace_defaults():
    context = TurnContext(
        turn_id="turn-1",
        session_id="session-1",
        character_id="tsundere_sister",
        player_input="师姐，帮帮我。",
    )

    trace = TurnTrace.start(context)
    trace.add_event(
        TurnEvent(
            event_type=EventType.TURN_STARTED,
            turn_id=context.turn_id,
            step_index=0,
            payload={"character_id": context.character_id},
        )
    )

    result = RuntimeResult(
        dialogue="先躲起来。",
        trace=trace,
        stop_reason="response_generated",
    )

    assert context.max_steps == 4
    assert StepBudget(max_steps=4).remaining_steps == 4
    assert trace.turn_id == "turn-1"
    assert trace.events[0].event_type is EventType.TURN_STARTED
    assert result.stop_reason == "response_generated"


def test_step_budget_consume_fails_when_budget_exhausted():
    budget = StepBudget(max_steps=1, used_steps=1)

    with pytest.raises(ValueError, match="Step budget exhausted"):
        budget.consume()


def test_step_budget_rejects_used_steps_above_max_at_construction():
    with pytest.raises(ValidationError):
        StepBudget(max_steps=1, used_steps=2)


def test_turn_trace_rejects_event_for_different_turn():
    context = TurnContext(
        turn_id="turn-1",
        session_id="session-1",
        character_id="tsundere_sister",
        player_input="师姐，帮帮我。",
    )
    trace = TurnTrace.start(context)

    with pytest.raises(ValueError, match="turn_id"):
        trace.add_event(
            TurnEvent(
                event_type=EventType.TURN_STARTED,
                turn_id="turn-2",
                step_index=0,
            )
        )


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (
            TurnContext,
            {
                "turn_id": "turn-1",
                "session_id": "session-1",
                "character_id": "tsundere_sister",
                "player_input": "师姐，帮帮我。",
                "unexpected": "x",
            },
        ),
        (
            StepBudget,
            {"max_steps": 4, "unexpected": "x"},
        ),
        (
            TurnEvent,
            {
                "event_type": EventType.TURN_STARTED,
                "turn_id": "turn-1",
                "unexpected": "x",
            },
        ),
        (
            TurnTrace,
            {
                "turn_id": "turn-1",
                "session_id": "session-1",
                "character_id": "tsundere_sister",
                "started_at": 0.0,
                "unexpected": "x",
            },
        ),
    ],
)
def test_contract_models_reject_unexpected_fields(factory, kwargs):
    with pytest.raises(ValidationError):
        factory(**kwargs)


def test_runtime_result_rejects_unexpected_fields():
    context = TurnContext(
        turn_id="turn-1",
        session_id="session-1",
        character_id="tsundere_sister",
        player_input="师姐，帮帮我。",
    )
    trace = TurnTrace.start(context)

    with pytest.raises(ValidationError):
        RuntimeResult(
            dialogue="先躲起来。",
            trace=trace,
            stop_reason="response_generated",
            unexpected="x",
        )
