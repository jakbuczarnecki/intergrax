# © Artur Czarnecki. All rights reserved.

"""GR-10-R8 — RuntimeToolInvoker canonical inner guard adoption."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
    CanonicalInnerGovernanceViolation,
)
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.audit import (
    GovernanceAuditRecorder,
    InMemoryGovernanceAuditSink,
)
from intergrax.runtime.agent_governance.capability_resolver import (
    InMemoryCapabilityGrantResolver,
)
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    AllowAllPolicyProvider,
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
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    ProductionRuntimeToolInvokerCompositionError,
    build_production_runtime_tool_invoker,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from tests.unit.runtime.architecture.gr3_inner_enforcement_ast import (
    collect_forbidden_concrete_inner_guard_imports,
    prepare_invocation_inner_guard_before_authorization_indices,
)
from tests.unit.runtime.governance.test_gr3_r1_explicit_inner_guard_composition import (
    _AllowingTestGuard,
    _RejectingTestGuard,
)
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_run_id_for_tests,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return _Out(value=request.input.value)


def _allow_all_governance() -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        tenant_id="test-tenant",
        agent_id="agent-1",
        allowed_capabilities=frozenset({"probe.tool"}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


class _RecordingGuard(CanonicalInnerExecutionGuardPort):
    def __init__(self, *, allow: bool) -> None:
        self.calls = 0
        self._allow = allow
        self.last_request: MeaningfulSideEffectRequest | None = None

    def assert_meaningful_side_effect_bound(self, request: MeaningfulSideEffectRequest) -> None:
        self.calls += 1
        self.last_request = request
        if not self._allow:
            raise CanonicalInnerGovernanceViolation(reason="test-deny")


def _contract() -> ToolContract:
    return ToolContract(
        tool_id="probe.tool",
        name="probe",
        description="probe",
        input_schema=_In,
        output_schema=_Out,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )


def _bind_test_governance_identity() -> object:
    return bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="test-tenant",
            workspace_id="test-workspace",
            principal_id="principal-A",
        ),
    )


def _invoke_with_identity(
    invoker: RuntimeToolInvoker,
    *,
    run_id: str,
    production_mode: bool = False,
) -> None:
    state = build_runtime_state_for_tests(run_id=run_id)
    if production_mode:
        state.context.config.production_mode = True
    request = ToolExecutionRequest(
        run_id=validate_run_id(run_id),
        tool_id="probe.tool",
        step_id="s1",
        input=_In(value=1),
    )
    run = validate_run_id(run_id)
    attempt = mint_attempt_id()
    execution = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run,
        attempt_id=attempt,
        execution_id=execution,
    )
    gov_token = _bind_test_governance_identity()
    try:
        invoker.invoke(state=state, agent_id="agent-1", request=request)
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(token)


def test_gr10_r8_custom_guard_deny_zero_physical_invocation() -> None:
    executor = _CountingExecutor()
    guard = _RecordingGuard(allow=False)
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
        inner_execution_guard=guard,
    )
    run_id = canonical_run_id_for_tests("r8-deny")
    try:
        _invoke_with_identity(invoker, run_id=run_id)
    except ToolGovernanceDeniedError:
        pass
    else:
        pytest.fail("expected ToolGovernanceDeniedError")
    assert guard.calls == 1
    assert executor.calls == 0


def test_gr10_r8_custom_guard_allow_single_physical_invocation() -> None:
    executor = _CountingExecutor()
    guard = _RecordingGuard(allow=True)
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
        inner_execution_guard=guard,
    )
    run_id = canonical_run_id_for_tests("r8-allow")
    _invoke_with_identity(invoker, run_id=run_id)
    assert guard.calls == 1
    assert executor.calls == 1
    assert guard.last_request is not None
    assert guard.last_request.action.endswith(":probe.tool")
    assert guard.last_request.principal_id == "principal-A"
    assert guard.last_request.principal_id != "agent-1"


def test_gr10_r8_production_mode_missing_guard_fail_closed() -> None:
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
    )
    run_id = canonical_run_id_for_tests("r8-missing-guard")
    state = build_runtime_state_for_tests(run_id=run_id)
    state.context.config.production_mode = True
    request = ToolExecutionRequest(
        run_id=validate_run_id(run_id),
        tool_id="probe.tool",
        step_id="s1",
        input=_In(value=1),
    )
    run = validate_run_id(run_id)
    token = bind_active_execution_identity(
        run_id=run,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    gov_token = _bind_test_governance_identity()
    try:
        try:
            invoker.invoke(state=state, agent_id="agent-1", request=request)
        except ToolGovernanceDeniedError as exc:
            assert exc.reason == "canonical_inner_execution_guard_not_configured"
        else:
            pytest.fail("expected ToolGovernanceDeniedError")
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(token)
    assert executor.calls == 0


def test_gr10_r8_production_composition_requires_governance() -> None:
    registry = FakeRegistry(_contract())
    with pytest.raises(ProductionRuntimeToolInvokerCompositionError):
        build_production_runtime_tool_invoker(
            registry=registry,
            production_mode=True,
        )


def test_gr10_r8_production_composition_wires_default_guard() -> None:
    from intergrax.runtime.governance.canonical_inner_execution_guard import (
        DefaultCanonicalInnerExecutionGuard,
    )

    registry = FakeRegistry(_contract())
    from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import (
        _RecordingMseBoundary,
    )

    from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
        materialize_tool_dependency_attempt_boundary,
    )
    from testing_support.dependency_concurrency_admission_config import (
        tool_dependency_concurrency_admission_configuration,
    )

    boundary = materialize_tool_dependency_attempt_boundary(
        tool_dependency_concurrency_admission_configuration("probe.tool", max_concurrent_calls=2),
        production_mode=True,
    )
    invoker = build_production_runtime_tool_invoker(
        registry=registry,
        agent_runtime_governance=_allow_all_governance(),
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        dependency_attempt_boundary=boundary,
        production_mode=True,
    )
    assert isinstance(invoker._inner_execution_guard, DefaultCanonicalInnerExecutionGuard)  # noqa: SLF001


def test_gr10_r8_invoker_has_no_concrete_inner_guard_import() -> None:
    tree = ast.parse(_INVOKER.read_text(encoding="utf-8-sig"), filename=str(_INVOKER))
    rel = _INVOKER.relative_to(_REPO_ROOT).as_posix()
    violations = collect_forbidden_concrete_inner_guard_imports(tree, rel_path=rel)
    assert violations == []


def test_gr10_r8_prepare_invocation_ast_call_order() -> None:
    tree = ast.parse(_INVOKER.read_text(encoding="utf-8-sig"), filename=str(_INVOKER))
    guard_idx, auth_idx = prepare_invocation_inner_guard_before_authorization_indices(tree)
    assert guard_idx is not None and auth_idx is not None
    assert guard_idx < auth_idx


def test_gr10_r8_allowing_and_rejecting_guards_are_swappable() -> None:
    executor = _CountingExecutor()
    allow_invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
        inner_execution_guard=_AllowingTestGuard(),
    )
    deny_invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
        inner_execution_guard=_RejectingTestGuard(),
    )
    run_id = canonical_run_id_for_tests("r8-swap")
    _invoke_with_identity(allow_invoker, run_id=run_id)
    assert executor.calls == 1
    executor.calls = 0
    try:
        _invoke_with_identity(deny_invoker, run_id=run_id)
    except ToolGovernanceDeniedError:
        pass
    else:
        pytest.fail("expected ToolGovernanceDeniedError")
    assert executor.calls == 0
