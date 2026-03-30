from __future__ import annotations

from typing import Any

from src.models.game_state import GameState
from src.observability.events import EventType, TurnEvent
from src.observability.trace import TurnTrace
from src.runtime.chat_policy import ChatRuntimePolicy
from src.runtime.policies import RuntimePolicy
from src.runtime.runtime_result import RuntimeResult
from src.runtime.step_budget import StepBudget
from src.runtime.stop_conditions import should_stop
from src.runtime.turn_context import TurnContext


class AgentRuntime:
    def __init__(
        self,
        policy: RuntimePolicy,
        tool_executor: Any | None = None,
        memory_adapter: Any | None = None,
        reflector: Any | None = None,
        action_planner: Any | None = None,
        perception: Any | None = None,
        character: Any | None = None,
        perception_context: Any | None = None,
        game_state: GameState | None = None,
    ):
        self.policy = policy
        self.tool_executor = tool_executor
        self.memory_adapter = memory_adapter
        self.reflector = reflector
        self.action_planner = action_planner
        self.perception = perception
        self.character = character
        self.perception_context = perception_context
        self.game_state = game_state

    def _build_initial_state(
        self,
        context: TurnContext,
        game_state: GameState | None = None,
    ) -> dict[str, Any]:
        state = {
            "context": context,
            "character_id": context.character_id,
            "game_state": game_state
            if game_state is not None
            else self.game_state
            if self.game_state is not None
            else GameState(session_id=context.session_id),
        }
        if self.memory_adapter is not None:
            state["memory_adapter"] = self.memory_adapter
        if self.reflector is not None:
            state["reflector"] = self.reflector
        if self.action_planner is not None:
            state["action_planner"] = self.action_planner
        if self.perception_context is not None:
            state["perception_context"] = self.perception_context
        return state

    def _validate_policy_dependencies(self) -> None:
        if not isinstance(self.policy, ChatRuntimePolicy):
            return

        missing = [
            name
            for name, value in (
                ("memory_adapter", self.memory_adapter),
                ("reflector", self.reflector),
                ("action_planner", self.action_planner),
            )
            if value is None
        ]
        if self.perception_context is None and (
            self.perception is None or self.character is None
        ):
            missing.append("perception_context")
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                "ChatRuntimePolicy requires these collaborators before run(): "
                f"{missing_text}"
            )

    def run(
        self,
        context: TurnContext,
        game_state: GameState | None = None,
    ) -> RuntimeResult:
        self._validate_policy_dependencies()
        trace = TurnTrace.start(context)
        budget = StepBudget(max_steps=context.max_steps)
        state = self._build_initial_state(context, game_state=game_state)
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

            if action.action_type == "retrieve_memory":
                if (
                    state.get("perception_context") is None
                    and self.perception is not None
                    and self.character is not None
                ):
                    state["perception_context"] = self.perception.perceive(
                        context.player_input,
                        self.character,
                        state["game_state"],
                    )
                    state["memory_snapshot"] = state[
                        "perception_context"
                    ].memory_result.model_dump()
                else:
                    state["memory_snapshot"] = state["memory_adapter"].retrieve(
                        query=context.player_input,
                        player_id=state["game_state"].player.player_id,
                    ).model_dump()
            elif action.action_type == "reflect":
                state["reflection"] = state["reflector"].generate(
                    context=state["perception_context"],
                )
            elif action.action_type == "plan_response":
                state["planned_response"] = state["action_planner"].plan(
                    context=state["perception_context"],
                    inner_monologue=state["reflection"],
                )
            elif action.action_type == "respond":
                dialogue = str(action.payload.get("dialogue", ""))
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
                continue
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
