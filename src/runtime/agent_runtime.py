from __future__ import annotations

from typing import Any

from src.models.game_state import GameState
from src.observability.events import EventType, TurnEvent
from src.observability.trace import TurnTrace
from src.runtime.policies import RuntimePolicy
from src.runtime.runtime_result import RuntimeResult
from src.runtime.step_budget import StepBudget
from src.runtime.stop_conditions import should_stop
from src.runtime.turn_context import TurnContext


class AgentRuntime:
    def __init__(self, policy: RuntimePolicy, tool_executor: Any | None = None):
        self.policy = policy
        self.tool_executor = tool_executor

    def run(self, context: TurnContext) -> RuntimeResult:
        trace = TurnTrace.start(context)
        budget = StepBudget(max_steps=context.max_steps)
        state: dict[str, Any] = {
            "context": context,
            "character_id": context.character_id,
            "game_state": GameState(session_id=context.session_id),
        }
        dialogue = ""
        stop_reason = "step_budget_exhausted"

        trace.add_event(
            TurnEvent(
                event_type=EventType.TURN_STARTED,
                turn_id=context.turn_id,
                step_index=0,
                payload={"character_id": context.character_id},
            )
        )

        while budget.remaining_steps > 0:
            step_index = budget.used_steps + 1
            trace.add_event(
                TurnEvent(
                    event_type=EventType.STEP_STARTED,
                    turn_id=context.turn_id,
                    step_index=step_index,
                )
            )

            action = self.policy.next_action(state)
            trace.add_event(
                TurnEvent(
                    event_type=EventType.ACTION_SELECTED,
                    turn_id=context.turn_id,
                    step_index=step_index,
                    payload={"action_type": action.action_type},
                )
            )

            if action.action_type == "respond":
                dialogue = str(action.payload.get("dialogue", ""))
            elif action.action_type == "propose_tool":
                tool_name = str(action.payload.get("tool_name", ""))
                tool_arguments = dict(action.payload.get("arguments") or {})
                executor = self.tool_executor
                if executor is None:
                    from src.adapters.tools.catalog import ToolCatalog
                    from src.adapters.tools.executor import ToolExecutor

                    executor = ToolExecutor(ToolCatalog.load())
                execution = executor.execute(tool_name, tool_arguments, state)
                state["last_tool_execution"] = execution

                trace.add_event(
                    TurnEvent(
                        event_type=EventType.STEP_COMPLETED,
                        turn_id=context.turn_id,
                        step_index=step_index,
                        payload={
                            "action_type": action.action_type,
                            **execution.as_event_payload(),
                        },
                    )
                )

                budget.consume()
                continue

            budget.consume()
            trace.add_event(
                TurnEvent(
                    event_type=EventType.STEP_COMPLETED,
                    turn_id=context.turn_id,
                    step_index=step_index,
                    payload={"action_type": action.action_type},
                )
            )

            stop_reason = should_stop(action.action_type, budget)
            if stop_reason is not None:
                break

        trace.add_event(
            TurnEvent(
                event_type=EventType.TURN_FINISHED,
                turn_id=context.turn_id,
                step_index=budget.used_steps,
                payload={"stop_reason": stop_reason, "dialogue": dialogue},
            )
        )
        trace.finish()
        return RuntimeResult(dialogue=dialogue, trace=trace, stop_reason=stop_reason)
