# © Artur Czarnecki. All rights reserved.

"""IntergraxAgent base class — thin UAEP authoring facade (Phase DX-2.3)."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from types import FunctionType
from typing import ClassVar, List

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.uaep_protocol import UAEPAgentWithDecide
from intergrax.agents.authoring.decisions import complete
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.task.task import TaskContext


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


class IntergraxAgent(HarnessReferenceAgent, UAEPAgentWithDecide, ABC):
    """
    Authoring base: declare ``contract_id``, ``capabilities``, implement ``@step`` methods.

    Subclasses must implement :meth:`build_context`.
    """

    contract_id: ClassVar[str] = "agent"
    capabilities: ClassVar[tuple[str, ...]] = ()
    agent_name: ClassVar[str] = "Intergrax Agent"
    agent_description: ClassVar[str] = "Authored Intergrax agent"
    agent_version: ClassVar[str] = "0.1.0"
    max_steps: ClassVar[int] = 10
    risk_level: ClassVar[AgentRiskLevel] = AgentRiskLevel.LOW
    skill_ids: ClassVar[tuple[str, ...]] = ()
    extra_tool_ids: ClassVar[tuple[str, ...]] = ()

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            capabilities=list(self.capabilities),
            skills=list(self.skill_ids),
            extra_tools=list(self.extra_tool_ids),
            risk_level=self.risk_level,
            max_steps=self.max_steps,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(self.capabilities)
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=list(supported),
                score=1.0,
                rationale="capability match",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def _step_methods(self) -> list[tuple[str, Callable[..., object]]]:
        methods: list[tuple[str, Callable[..., object]]] = []
        for cls in type(self).mro():
            if cls is IntergraxAgent or cls is object:
                continue
            for name, value in cls.__dict__.items():
                if not callable(value):
                    continue
                if _step_id_on_callable(value) is not None:
                    methods.append((name, value))
        methods.sort(key=lambda item: _step_id_on_callable(item[1]) or "")
        return methods

    def _ordered_step_ids(self) -> list[str]:
        ids: list[str] = []
        for _name, method in self._step_methods():
            step_id = _step_id_on_callable(method)
            if step_id is not None:
                ids.append(step_id)
        return ids

    def get_steps(self, context: RuntimeContext) -> List[AgentStep]:
        _ = context
        contract = self.get_contract()
        steps: list[AgentStep] = []
        for index, (method_name, method) in enumerate(self._step_methods()):
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

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        for _name, method in self._step_methods():
            if _step_id_on_callable(method) != step.step_id:
                continue
            result = await method(self, ctx)
            if isinstance(result, StepOutput):
                return result
            if isinstance(result, dict):
                summary = result.get("summary", "")
                return StepOutput(
                    step_id=step.step_id,
                    summary=str(summary),
                    data={k: v for k, v in result.items() if k != "summary"},
                )
            return StepOutput(step_id=step.step_id, summary=str(result))
        raise KeyError(f"Unknown step_id: {step.step_id}")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = output, ctx
        step_ids = self._ordered_step_ids()
        if not step_ids or step.step_id == step_ids[-1]:
            return complete(reason=f"{step.step_id} finished")
        index = step_ids.index(step.step_id)
        return AgentDecision(
            type=AgentDecisionType.CONTINUE,
            next_step_id=step_ids[index + 1],
            reason="next authored step",
        )
