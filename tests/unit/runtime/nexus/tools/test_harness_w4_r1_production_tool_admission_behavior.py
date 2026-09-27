# © Artur Czarnecki. All rights reserved.

"""HARNESS-W4-R1 — production composition behavioral admission proofs."""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicyMissingError,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    materialize_tool_dependency_attempt_boundary,
)
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.core.contracts import ToolContract, ToolRetryPolicy
from intergrax.tools.execution_models import (
    ToolEffectCertainty,
    ToolExecutionRequest,
)
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_run_id_for_tests,
)
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)
from tests.unit.runtime.governance.test_gr3_r1_explicit_inner_guard_composition import (
    _AllowingTestGuard,
)
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.runtime.agent_governance.audit import (
    GovernanceAuditRecorder,
    InMemoryGovernanceAuditSink,
)
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.capability_resolver import (
    InMemoryCapabilityGrantResolver,
)
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    AllowAllPolicyProvider,
)
from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import (
    _RecordingMseBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PROBE_TOOL = "probe.tool"
_OTHER_TOOL = "probe.other"


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    result: int


class _Handler:
    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        return _Out(result=request.input.value)


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self._gate: threading.Event | None = None
        self._lock = threading.Lock()

    def set_gate(self, gate: threading.Event) -> None:
        self._gate = gate

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        with self._lock:
            self.calls += 1
        if self._gate is not None:
            self._gate.wait(timeout=10)
        return _Out(result=request.input.value)


def _register(registry: ToolRegistry, tool_id: str) -> None:
    registry.register(
        contract=ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=tool_id,
            input_schema=_In,
            output_schema=_Out,
            error_mapping={},
            side_effects=False,
            retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
        ),
        handler=_Handler(),
    )


def _governance_for_tool_capabilities(*capabilities: str) -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        tenant_id="test-tenant",
        agent_id="agent-1",
        allowed_capabilities=frozenset(capabilities),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


def _production_invoker(
    registry: ToolRegistry,
    executor: _CountingExecutor,
    *,
    policy_tool_id: str,
    max_concurrent_calls: int = 1,
    overload_mode: DependencyConcurrencyOverloadMode = DependencyConcurrencyOverloadMode.REJECT,
    wait_timeout_seconds: float | None = None,
    allowed_tools: set[str] | None = None,
) -> RuntimeToolInvoker:
    config = tool_dependency_concurrency_admission_configuration(
        policy_tool_id,
        max_concurrent_calls=max_concurrent_calls,
        overload_mode=overload_mode,
        wait_timeout_seconds=wait_timeout_seconds,
    )
    boundary = materialize_tool_dependency_attempt_boundary(config, production_mode=True)
    assert boundary is not None
    scope = allowed_tools or {policy_tool_id}
    return build_production_runtime_tool_invoker(
        registry=registry,
        executor=executor,
        dependency_attempt_boundary=boundary,
        production_mode=True,
        agent_runtime_governance=_governance_for_tool_capabilities(policy_tool_id),
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        inner_execution_guard=_AllowingTestGuard(),
        scope_policy=StaticToolScopePolicy(allowed_tools=scope),
    )


def _state(run_id: str):
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="w4r1.admission")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    state = build_runtime_state_for_tests(run_id=run_id)
    state.context.config.policy_bundle = wire_policy_bundle(env)
    return state


def _identity_scope(run_id: str):
    run = validate_run_id(run_id)
    exec_token = bind_active_execution_identity(
        run_id=run,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="test-tenant",
            workspace_id="test-workspace",
            principal_id="principal-A",
        ),
    )
    return exec_token, gov_token


def _reset_identity(exec_token: object, gov_token: object) -> None:
    reset_active_execution_governance_identity(gov_token)
    reset_active_execution_identity(exec_token)


def test_w4_r1_reject_second_invocation_without_physical_executor_call() -> None:
    registry = ToolRegistry()
    _register(registry, _PROBE_TOOL)
    executor = _CountingExecutor()
    invoker = _production_invoker(registry, executor, policy_tool_id=_PROBE_TOOL)
    gate = threading.Event()
    executor.set_gate(gate)
    run_id = canonical_run_id_for_tests("w4r1-reject")
    holder_state = _state(run_id)
    state = _state(run_id)

    def _hold() -> None:
        exec_token, gov_token = _identity_scope(run_id)
        try:
            invoker.invoke(
                state=holder_state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id),
                    tool_id=_PROBE_TOOL,
                    step_id="1",
                    input=_In(value=1),
                ),
            )
        finally:
            _reset_identity(exec_token, gov_token)

    holder = threading.Thread(target=_hold)
    holder.start()
    time.sleep(0.15)
    exec_token, gov_token = _identity_scope(run_id)
    try:
        result = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id=_PROBE_TOOL,
                step_id="2",
                input=_In(value=2),
            ),
        )
    finally:
        _reset_identity(exec_token, gov_token)
    gate.set()
    holder.join(timeout=10)
    invoker.close()
    assert executor.calls == 1
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.DEPENDENCY_ERROR
    assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED


def test_w4_r1_permit_release_allows_subsequent_invocation() -> None:
    registry = ToolRegistry()
    _register(registry, _PROBE_TOOL)
    executor = _CountingExecutor()
    invoker = _production_invoker(registry, executor, policy_tool_id=_PROBE_TOOL)
    run_id = canonical_run_id_for_tests("w4r1-release")
    state = _state(run_id)
    exec_token, gov_token = _identity_scope(run_id)
    try:
        first = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id=_PROBE_TOOL,
                step_id="1",
                input=_In(value=2),
            ),
        )
        second = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id=_PROBE_TOOL,
                step_id="2",
                input=_In(value=3),
            ),
        )
    finally:
        _reset_identity(exec_token, gov_token)
    invoker.close()
    assert first.success and second.success
    assert executor.calls == 2


def test_w4_r1_wait_with_timeout_rejects_without_physical_start() -> None:
    registry = ToolRegistry()
    _register(registry, _PROBE_TOOL)
    executor = _CountingExecutor()
    invoker = _production_invoker(
        registry,
        executor,
        policy_tool_id=_PROBE_TOOL,
        max_concurrent_calls=1,
        overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=0.05,
    )
    gate = threading.Event()
    executor.set_gate(gate)
    run_id = canonical_run_id_for_tests("w4r1-timeout")
    holder_state = _state(run_id)
    state = _state(run_id)

    def _hold() -> None:
        exec_token, gov_token = _identity_scope(run_id)
        try:
            invoker.invoke(
                state=holder_state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id),
                    tool_id=_PROBE_TOOL,
                    step_id="1",
                    input=_In(value=3),
                ),
            )
        finally:
            _reset_identity(exec_token, gov_token)

    holder = threading.Thread(target=_hold)
    holder.start()
    time.sleep(0.15)
    exec_token, gov_token = _identity_scope(run_id)
    try:
        result = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id=_PROBE_TOOL,
                step_id="2",
                input=_In(value=4),
            ),
        )
    finally:
        _reset_identity(exec_token, gov_token)
    gate.set()
    holder.join(timeout=10)
    invoker.close()
    assert executor.calls == 1
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.DEPENDENCY_ERROR
    assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED


def test_w4_r1_missing_tool_policy_fails_closed() -> None:
    registry = ToolRegistry()
    _register(registry, _PROBE_TOOL)
    _register(registry, _OTHER_TOOL)
    executor = _CountingExecutor()
    invoker = build_production_runtime_tool_invoker(
        registry=registry,
        executor=executor,
        dependency_attempt_boundary=materialize_tool_dependency_attempt_boundary(
            tool_dependency_concurrency_admission_configuration(
                _PROBE_TOOL,
                max_concurrent_calls=2,
            ),
            production_mode=True,
        ),
        production_mode=True,
        agent_runtime_governance=_governance_for_tool_capabilities(
            _PROBE_TOOL,
            _OTHER_TOOL,
        ),
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        inner_execution_guard=_AllowingTestGuard(),
        scope_policy=StaticToolScopePolicy(allowed_tools={_PROBE_TOOL, _OTHER_TOOL}),
    )
    run_id = canonical_run_id_for_tests("w4r1-missing-policy")
    state = _state(run_id)
    exec_token, gov_token = _identity_scope(run_id)
    try:
        with pytest.raises(DependencyConcurrencyPolicyMissingError):
            invoker.invoke(
                state=state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id),
                    tool_id=_OTHER_TOOL,
                    step_id="1",
                    input=_In(value=4),
                ),
            )
    finally:
        _reset_identity(exec_token, gov_token)
    invoker.close()
    assert executor.calls == 0
