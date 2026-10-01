# © Artur Czarnecki. All rights reserved.

"""EBH-4-R1-R3-B6-R2 — runtime contract conformance (closed-world)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.agent_execution_validation_engine import (
    AgentExecutionValidationEnginePort,
)
from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionDecision,
    DecisionExposureSelectionHostBinding,
    DecisionExposureSelectionStrategy,
    DecisionExposureSelectionSuccessReason,
    HostPublicationClass,
)
from intergrax.contracts.decision_authoritative_exposure import DecisionEvaluationScope
from intergrax.contracts.decision_identity import DecisionExecutionLineage, DecisionScope
from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationMiddlewarePipelinePort,
    HostOrchestrationRuntimeEventPort,
    HostOrchestrationRuntimeMiddlewareRegistration,
)
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.plugins.bootstrap import (
    RuntimePluginPolicyRegistrationUnsupportedError,
    bootstrap_runtime_plugins,
)
from intergrax.runtime.plugins.contract import PolicyEngineLike, RuntimePlugin
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _CustomValidationEngine:
    def validate(
        self,
        execution: AgentExecutionResult,
        *,
        contract: AgentContract,
        capability: str | None = None,
        plan_criteria: list[str] | None = None,
    ) -> ValidationResult:
        _ = (execution, contract, capability, plan_criteria)
        return ValidationResult(valid=True, errors=[])


def _accept_validation_port(value: AgentExecutionValidationEnginePort) -> None:
    _ = value


def _accept_exposure_binding(value: DecisionExposureSelectionHostBinding) -> None:
    _ = value


@dataclass(frozen=True, slots=True)
class _FakeExposureStrategy:
    strategy_id: str = "conformance.fake.strategy"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object]:
        _ = policy
        if not candidates:
            raise AssertionError("expected candidates")
        return DecisionExposureSelectionDecision(
            selected=candidates[0].exposure,
            reason_code=DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE,
            considered_candidates=len(candidates),
        )


@dataclass(frozen=True, slots=True)
class _FakeExposureBinding:
    strategy: DecisionExposureSelectionStrategy[object]


def test_custom_validation_port_accepted_by_nexus_wiring() -> None:
    custom = _CustomValidationEngine()
    assert not isinstance(custom, NexusValidationEngine)
    _accept_validation_port(custom)
    loop = NexusLoop(AgentRegistry())
    loop.apply_validation_engine(custom)
    assert loop._validation_engine is custom  # noqa: SLF001
    executor = GraphExecutor(AgentRegistry())
    executor.apply_validation_engine(custom)
    assert executor._validation_engine is custom  # noqa: SLF001


def test_custom_exposure_binding_accepted_without_composition_identity() -> None:
    binding = _FakeExposureBinding(strategy=_FakeExposureStrategy())
    _accept_exposure_binding(binding)
    loop = NexusLoop(AgentRegistry())
    loop.apply_decision_exposure_selection(binding)
    assert loop._decision_exposure_selection is binding  # noqa: SLF001


@dataclass(frozen=True)
class _FakePolicyRule:
    rule_id: str = "plugin.fake.rule"


def test_plugin_policy_registration_is_explicitly_unsupported() -> None:
    def _attempt_register(
        _event_bus: object,
        _hooks: object,
        policy: PolicyEngineLike,
    ) -> None:
        policy.register_rule(_FakePolicyRule())

    plugins = [
        RuntimePlugin(
            plugin_id="conformance.policy",
            version="1.0.0",
            register=_attempt_register,
        ),
    ]
    with pytest.raises(RuntimePluginPolicyRegistrationUnsupportedError):
        bootstrap_runtime_plugins(
            plugins,
            event_bus=RuntimeEventBus(),
            hook_registry=MiddlewarePipeline().hooks,
        )


class _CountingMiddleware(RuntimeMiddleware):
    name = "conformance.counting"
    priority = 50

    def __init__(self) -> None:
        self.before_calls = 0
        self.after_calls = 0

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        _ = (point, ctx)
        self.before_calls += 1
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        _ = (point, ctx)
        self.after_calls += 1
        return HookResult()


def test_middleware_pipeline_structural_host_port_conformance() -> None:
    pipeline = MiddlewarePipeline()
    assert isinstance(pipeline, HostOrchestrationMiddlewarePipelinePort)
    assert isinstance(RuntimeEventBus(), HostOrchestrationRuntimeEventPort)
    mw = _CountingMiddleware()
    assert isinstance(mw, HostOrchestrationRuntimeMiddlewareRegistration)
    pipeline.attach_runtime_middleware_if_absent(mw)
    pipeline.attach_runtime_middleware_if_absent(mw)
    names = pipeline.registered_middleware_names()
    assert names == frozenset({"conformance.counting"})
