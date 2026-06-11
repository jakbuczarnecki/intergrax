# © Artur Czarnecki. All rights reserved.

"""Internal UAEP bridge for linear ``@step`` agents — not author API (ACP-CLOSE-LEG-2)."""

from __future__ import annotations

from collections.abc import Callable
from types import FunctionType
from typing import TYPE_CHECKING

from intergrax.agents.authoring.decisions import complete
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext

if TYPE_CHECKING:
    from intergrax.agents.authoring.base import IntergraxAgent


def _step_id_on_callable(value: object) -> str | None:
    if not isinstance(value, FunctionType):
        return None
    raw = value.__dict__.get("__intergrax_step_id__")
    return raw if isinstance(raw, str) else None


def _trace_label_on_callable(value: object, step_id: str) -> str:
    if isinstance(value, FunctionType):
        raw = value.__dict__.get("__intergrax_trace_label__")
        if isinstance(raw, str) and raw:
            return raw
    return step_id


def linear_step_methods(agent: IntergraxAgent) -> list[tuple[str, Callable[..., object]]]:
    methods: list[tuple[str, Callable[..., object]]] = []
    for cls in type(agent).mro():
        if cls.__name__ == "IntergraxAgent" or cls is object:
            continue
        for name, value in cls.__dict__.items():
            if not callable(value):
                continue
            if _step_id_on_callable(value) is not None:
                methods.append((name, value))
    methods.sort(key=lambda item: _step_id_on_callable(item[1]) or "")
    return methods


def linear_ordered_step_ids(agent: IntergraxAgent) -> list[str]:
    ids: list[str] = []
    for _name, method in linear_step_methods(agent):
        step_id = _step_id_on_callable(method)
        if step_id is not None:
            ids.append(step_id)
    return ids


def linear_agent_get_steps(agent: IntergraxAgent, context: RuntimeContext) -> list[AgentStep]:
    _ = context
    contract = agent.get_contract()
    steps: list[AgentStep] = []
    for index, (method_name, method) in enumerate(linear_step_methods(agent)):
        step_id = _step_id_on_callable(method)
        if step_id is None:
            continue
        steps.append(
            AgentStep(
                step_id=step_id,
                step_name=method_name,
                step_index=index,
                trace_label=_trace_label_on_callable(method, step_id),
                allowed_tools=list(contract.allowed_tools),
            )
        )
    return steps


def linear_agent_decide_after_step(
    agent: IntergraxAgent,
    step: AgentStep,
    output: StepOutput | None,
    ctx: RuntimeExecutionContext,
) -> AgentDecision:
    _ = output, ctx
    step_ids = linear_ordered_step_ids(agent)
    if not step_ids or step.step_id == step_ids[-1]:
        return complete(reason=f"{step.step_id} finished")
    index = step_ids.index(step.step_id)
    return AgentDecision(
        type=AgentDecisionType.CONTINUE,
        reason="next authored step",
        payload={"next_step_id": step_ids[index + 1]},
    )
