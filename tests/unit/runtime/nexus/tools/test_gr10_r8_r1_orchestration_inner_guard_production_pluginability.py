# © Artur Czarnecki. All rights reserved.

"""GR-10-R8-R1 — production inner guard pluginability and canonical principal projection."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
    CanonicalInnerGovernanceViolation,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
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
from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    DenyCapabilityPolicyProvider,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.canonical_inner_execution_guard import (
    DefaultCanonicalInnerExecutionGuard,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_inner_governance import (
    build_tool_invocation_inner_governance_request,
)
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
    default_lab_capability_grants,
)
from intergrax.runtime.wiring.harness_governance import create_lab_allow_governance_service
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolProfile, ToolWiringContext
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from testing_support.builder import (
    FakeLLMAdapter,
    build_runtime_state_for_tests,
    canonical_run_id_for_tests,
)
from tests.unit.runtime.architecture.gr3_inner_enforcement_ast import (
    collect_forbidden_concrete_inner_guard_imports,
)
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _CountingExecutor,
    _RecordingGuard,
    _allow_all_governance,
    _bind_test_governance_identity,
    _contract,
    _invoke_with_identity,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIG = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "config.py"


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


class _CustomProductionGuard(CanonicalInnerExecutionGuardPort):
    def __init__(self) -> None:
        self.calls = 0

    def assert_meaningful_side_effect_bound(self, request: MeaningfulSideEffectRequest) -> None:
        self.calls += 1


class _CountingGovernancePort:
    def __init__(self, inner: AgentRuntimeGovernancePort, order: list[str]) -> None:
        self._inner = inner
        self._order = order
        self.calls = 0

    def authorize_tool(self, request):  # noqa: ANN001
        self.calls += 1
        self._order.append("tool_governance")
        return self._inner.authorize_tool(request)


class _OrderExecutor:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        self._order.append("executor")
        return _Out(value=request.input.value)


class _OrderRecordingGuard(CanonicalInnerExecutionGuardPort):
    def __init__(self, order: list[str], *, allow: bool) -> None:
        self._order = order
        self._allow = allow

    def assert_meaningful_side_effect_bound(self, request: MeaningfulSideEffectRequest) -> None:
        self._order.append("inner_guard")
        if not self._allow:
            raise CanonicalInnerGovernanceViolation(reason="test-inner-deny")


def _deny_tool_governance() -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        tenant_id="test-tenant",
        agent_id="agent-1",
        allowed_capabilities=frozenset({"probe.tool"}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine(
            (DenyCapabilityPolicyProvider(denied_capabilities=frozenset({"probe.tool"})),),
        ),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


def test_gr10_r8_r1_runtime_config_exposes_guard_port_without_concrete_import() -> None:
    tree = ast.parse(_CONFIG.read_text(encoding="utf-8-sig"), filename=str(_CONFIG))
    rel = _CONFIG.relative_to(_REPO_ROOT).as_posix()
    assert collect_forbidden_concrete_inner_guard_imports(tree, rel_path=rel) == []
    source = _CONFIG.read_text(encoding="utf-8-sig")
    assert "canonical_inner_execution_guard" in source
    assert "DefaultCanonicalInnerExecutionGuard" not in source


def test_gr10_r8_r1_runtime_context_custom_guard_end_to_end() -> None:
    custom = _CustomProductionGuard()
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
        production_mode=True,
        trace_db_path="/tmp/trace.db",
        tool_registry=FakeRegistry(_contract()),
        agent_runtime_governance=_allow_all_governance(),
        canonical_inner_execution_guard=custom,
    )
    ctx = RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
        governance_service=create_lab_allow_governance_service(),
    )
    invoker = ctx.config.tool_invoker
    assert invoker is not None
    _invoke_with_identity(
        invoker,
        run_id=canonical_run_id_for_tests("r8r1-ctx"),
        production_mode=True,
    )
    assert custom.calls == 1
    assert not isinstance(invoker._inner_execution_guard, DefaultCanonicalInnerExecutionGuard)  # noqa: SLF001


def test_gr10_r8_r1_runtime_context_default_guard_when_custom_absent() -> None:
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
        production_mode=True,
        trace_db_path="/tmp/trace.db",
        tool_registry=FakeRegistry(_contract()),
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=default_lab_capability_grants("test-tenant"),
        ),
    )
    ctx = RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
        governance_service=create_lab_allow_governance_service(),
    )
    invoker = ctx.config.tool_invoker
    assert invoker is not None
    assert isinstance(invoker._inner_execution_guard, DefaultCanonicalInnerExecutionGuard)  # noqa: SLF001


def test_gr10_r8_r1_declarative_wiring_custom_guard_end_to_end() -> None:
    custom = _CustomProductionGuard()
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["read_file"]),
        wiring_context=ToolWiringContext(),
        registry=FakeRegistry(_contract()),
    )
    catalog = build_declarative_invoker_from_tool_wiring(
        wiring,
        agent_runtime_governance=_allow_all_governance(),
        canonical_inner_execution_guard=custom,
        production_mode=True,
    )
    assert catalog is not None
    _invoke_with_identity(
        catalog.tool_invoker,
        run_id=canonical_run_id_for_tests("r8r1-decl"),
        production_mode=True,
    )
    assert custom.calls == 1


def test_gr10_r8_r1_principal_projection_separates_agent_and_governance_principal() -> None:
    state = build_runtime_state_for_tests(run_id=canonical_run_id_for_tests("r8r1-principal"))
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="test-tenant",
            workspace_id="test-workspace",
            principal_id="principal-A",
        ),
    )
    exec_token = bind_active_execution_identity(
        run_id=validate_run_id(state.run_id),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        request = build_tool_invocation_inner_governance_request(
            state=state,
            agent_id="agent-B",
            contract=_contract(),
            request=ToolExecutionRequest(
                run_id=validate_run_id(state.run_id),
                tool_id="probe.tool",
                step_id="s1",
                input=_In(value=1),
            ),
        )
    finally:
        reset_active_execution_identity(exec_token)
        reset_active_execution_governance_identity(gov_token)
    assert request.principal_id == "principal-A"
    assert request.principal_id != "agent-B"
    assert request.tenant_id == "test-tenant"


def test_gr10_r8_r1_behavioral_invoke_order_allow() -> None:
    order: list[str] = []
    guard = _OrderRecordingGuard(order, allow=True)
    inner_gov = _CountingGovernancePort(_allow_all_governance(), order)
    executor = _OrderExecutor(order)
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
        inner_execution_guard=guard,
        agent_runtime_governance=inner_gov,
    )
    _invoke_with_identity(
        invoker,
        run_id=canonical_run_id_for_tests("r8r1-order-allow"),
        production_mode=True,
    )
    assert order == ["inner_guard", "tool_governance", "executor"]


def test_gr10_r8_r1_behavioral_inner_deny_blocks_governance_and_executor() -> None:
    order: list[str] = []
    guard = _OrderRecordingGuard(order, allow=False)
    inner_gov = _CountingGovernancePort(_allow_all_governance(), order)
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
        inner_execution_guard=guard,
        agent_runtime_governance=inner_gov,
    )
    try:
        _invoke_with_identity(
            invoker,
            run_id=canonical_run_id_for_tests("r8r1-inner-deny"),
            production_mode=True,
        )
    except ToolGovernanceDeniedError:
        pass
    else:
        pytest.fail("expected ToolGovernanceDeniedError")
    assert order == ["inner_guard"]
    assert inner_gov.calls == 0
    assert executor.calls == 0


def test_gr10_r8_r1_behavioral_tool_governance_deny_blocks_executor() -> None:
    order: list[str] = []
    guard = _OrderRecordingGuard(order, allow=True)
    inner_gov = _CountingGovernancePort(_deny_tool_governance(), order)
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(_contract()),
        executor=executor,
        inner_execution_guard=guard,
        agent_runtime_governance=inner_gov,
    )
    try:
        _invoke_with_identity(
            invoker,
            run_id=canonical_run_id_for_tests("r8r1-gov-deny"),
            production_mode=True,
        )
    except ToolGovernanceDeniedError:
        pass
    else:
        pytest.fail("expected ToolGovernanceDeniedError")
    assert "inner_guard" in order
    assert "tool_governance" in order
    assert executor.calls == 0
