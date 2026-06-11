# © Artur Czarnecki. All rights reserved.

"""IntergraxAgent base class — thin UAEP authoring facade (Phase DX-2.3)."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from types import FunctionType
from typing import ClassVar, List

from intergrax.agents.authoring.acp_run import run_acp_session
from intergrax.agents.authoring.decisions import complete
from intergrax.agents.authoring.state_access import load_session_state, session_state_delta
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment
from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.uaep_protocol import UAEPAgentWithDecide
from intergrax.contracts.acp_state import AcpSessionState
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_contract_section12 import (
    DEFAULT_FAILURE_MODES,
    DEFAULT_INPUT_SCHEMA,
    DEFAULT_OUTPUT_SCHEMA,
    DEFAULT_VALIDATION_RULES,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.agent_run import AgentRunError, AgentRunRequest, AgentRunResult
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, AgentRunStatus, TerminalReason
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
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
    session_state_type: ClassVar[type[AcpSessionState]] = AcpSessionState

    async def run(self, request: AgentRunRequest | RuntimeRequest) -> AgentRunResult | RuntimeAnswer:
        if isinstance(request, AgentRunRequest):
            return await run_acp_session(self, request)
        from intergrax.agents.agent_engine import AgentEngine

        return await AgentEngine.run_agent(self, request)

    def configure_run(self, merged: EffectiveAgentRunEnvironment) -> dict[str, object]:
        """Per-run domain overlay merged into session metadata (§29.5)."""
        _ = merged
        return {}

    async def on_run_start(self, merged: EffectiveAgentRunEnvironment) -> None:
        _ = merged

    async def on_run_end(self, result: AgentRunResult) -> None:
        _ = result

    def validate_output(self, result: AgentRunResult) -> ValidationResult:
        if result.status == AgentRunStatus.FAILED:
            return ValidationResult(valid=False, errors=[error.message for error in result.errors])
        if result.output in ("", None, {}):
            return ValidationResult(valid=False, errors=["empty output"])
        return ValidationResult(valid=True)

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            capabilities=list(self.capabilities),
            skills=list(self.skill_ids),
            extra_tools=list(self.extra_tool_ids),
            input_schema=dict(DEFAULT_INPUT_SCHEMA),
            output_schema=dict(DEFAULT_OUTPUT_SCHEMA),
            validation_rules=list(DEFAULT_VALIDATION_RULES),
            failure_modes=list(DEFAULT_FAILURE_MODES),
            risk_level=self.risk_level,
            max_steps=self.max_steps,
        )

    def load_session_state(self, step_ctx: AgentStepContext) -> AcpSessionState:
        return load_session_state(step_ctx, state_type=self.session_state_type)

    def session_state_delta(
        self,
        model: AcpSessionState,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        exclude_none: bool = True,
    ) -> dict[str, object]:
        return session_state_delta(
            model,
            include=include,
            exclude=exclude,
            exclude_none=exclude_none,
        )

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        """
        Primary cognitive hook — override for custom loops.

        Default drives authored ``@step`` methods when ``uaep_exec_ctx`` is present
        in ``step_ctx.metadata`` (internal bridge until ACP-STEP-3).
        """
        step_ids = self._ordered_step_ids()
        if not step_ids:
            return StepOutcome.fail(
                [
                    AgentRunError(
                        code=AgentRunErrorCode.VALIDATION_FAILED,
                        message="no authored steps registered",
                    )
                ],
                terminal_reason=TerminalReason.VALIDATION_FAILED,
            )
        if step_ctx.step_index >= len(step_ids):
            return StepOutcome.complete(
                output={"status": "complete"},
                terminal_reason=TerminalReason.GOAL_MET,
            )

        exec_ctx = step_ctx.metadata.get("uaep_exec_ctx")
        if not isinstance(exec_ctx, RuntimeExecutionContext):
            return StepOutcome.continue_with(
                state_delta={"iteration": step_ctx.step_index + 1},
            )

        step_id = step_ids[step_ctx.step_index]
        step = AgentStep(
            step_id=step_id,
            step_name=step_id,
            step_index=step_ctx.step_index,
        )
        output = await self.run_step(step, exec_ctx)
        if step_ctx.step_index >= len(step_ids) - 1:
            return StepOutcome.complete(
                output=output.model_dump(),
                terminal_reason=TerminalReason.GOAL_MET,
            )
        return StepOutcome.continue_with(
            state_delta={"iteration": step_ctx.step_index + 1},
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
