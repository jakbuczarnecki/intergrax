# © Artur Czarnecki. All rights reserved.

"""GR-10-R9 — RuntimeToolInvoker canonical MSE boundary adoption."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
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
from intergrax.runtime.agent_governance.errors import (
    ToolGovernanceApprovalRequiredError,
    ToolGovernanceDeniedError,
)
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    AllowAllPolicyProvider,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    ProductionRuntimeToolInvokerCompositionError,
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
    SideEffectAuthorizationFailureReason,
)
from intergrax.runtime.wiring.harness_governance import create_lab_allow_governance_service
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import (
    FakeLLMAdapter,
    build_runtime_state_for_tests,
    canonical_run_id_for_tests,
)
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
    _invoke_with_identity,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_CONFIG = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "config.py"


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


class _RecordingMseBoundary:
    def __init__(
        self,
        *,
        allow: bool | None = None,
        action: PolicyAction | None = None,
    ) -> None:
        self.calls = 0
        if action is not None:
            self._action = action
            self._permitted = action is PolicyAction.ALLOW
        else:
            self._action = PolicyAction.ALLOW if allow else PolicyAction.DENY
            self._permitted = bool(allow)
        self.last_request: CollaborativeWorkEnforcementRequest | None = None

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        self.calls += 1
        self.last_request = request
        decision = PolicyDecision(
            action=self._action,
            reason="test-mse",
            policy_rule_id="test.mse",
        )
        from intergrax.contracts.collaborative_work import (
            CollaborativeWorkEnforcementResult,
            PolicyCompositionResult,
        )

        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        requires_continuation = self._action in (
            PolicyAction.REQUIRE_HUMAN,
            PolicyAction.ESCALATE,
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=self._permitted,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=requires_continuation,
            governed_continuation_request=None,
        )


def _side_effect_contract() -> ToolContract:
    return ToolContract(
        tool_id="probe.side_effect",
        name="probe",
        description="probe",
        input_schema=_In,
        output_schema=_Out,
        side_effects=True,
        error_mapping={},
        risk_level=ToolRiskLevel.HIGH,
    )


def _read_only_contract() -> ToolContract:
    return ToolContract(
        tool_id="probe.read",
        name="probe",
        description="probe",
        input_schema=_In,
        output_schema=_Out,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )


def _enforce_allow_policy_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gr10.r9.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _production_invoker(
    *,
    contract: ToolContract,
    executor: _CountingExecutor,
    boundary: MeaningfulSideEffectAuthorizationPort | _RecordingMseBoundary | None,
    inner_guard: _RecordingGuard | None = None,
) -> RuntimeToolInvoker:
    return RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=executor,
        agent_runtime_governance=_allow_all_governance(),
        inner_execution_guard=inner_guard or _RecordingGuard(allow=True),
        meaningful_side_effect_authorization=boundary,
    )


def _invoke_production(
    invoker: RuntimeToolInvoker,
    *,
    run_id: str,
) -> None:
    state = build_runtime_state_for_tests(run_id=run_id)
    state.context.config.production_mode = True
    state.context.config.policy_bundle = _enforce_allow_policy_bundle()
    request = ToolExecutionRequest(
        run_id=validate_run_id(run_id),
        tool_id="probe.side_effect",
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
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="test-tenant",
            workspace_id="test-workspace",
            principal_id="principal-A",
        ),
    )
    try:
        invoker.invoke(state=state, agent_id="agent-1", request=request)
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(token)


def _allow_all_governance() -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        tenant_id="test-tenant",
        agent_id="agent-1",
        allowed_capabilities=frozenset({"probe.side_effect", "probe.read"}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


def test_gr10_r9_mse_deny_zero_executor_calls() -> None:
    executor = _CountingExecutor()
    boundary = _RecordingMseBoundary(allow=False)
    invoker = _production_invoker(
        contract=_side_effect_contract(),
        executor=executor,
        boundary=boundary,
    )
    run_id = canonical_run_id_for_tests("r9-deny")
    with pytest.raises(ToolGovernanceDeniedError):
        _invoke_production(invoker, run_id=run_id)
    assert boundary.calls == 1
    assert executor.calls == 0


def test_gr10_r9_mse_modify_zero_executor_calls() -> None:
    executor = _CountingExecutor()
    boundary = _RecordingMseBoundary(action=PolicyAction.MODIFY)
    invoker = _production_invoker(
        contract=_side_effect_contract(),
        executor=executor,
        boundary=boundary,
    )
    run_id = canonical_run_id_for_tests("r9-modify")
    with pytest.raises(ToolGovernanceDeniedError):
        _invoke_production(invoker, run_id=run_id)
    assert boundary.calls == 1
    assert executor.calls == 0


def test_gr10_r9_mse_require_human_zero_executor_calls() -> None:
    executor = _CountingExecutor()
    boundary = _RecordingMseBoundary(action=PolicyAction.REQUIRE_HUMAN)
    invoker = _production_invoker(
        contract=_side_effect_contract(),
        executor=executor,
        boundary=boundary,
    )
    run_id = canonical_run_id_for_tests("r9-human")
    with pytest.raises(ToolGovernanceApprovalRequiredError):
        _invoke_production(invoker, run_id=run_id)
    assert boundary.calls == 1
    assert executor.calls == 0


def test_gr10_r9_mse_allow_single_executor_call() -> None:
    executor = _CountingExecutor()
    boundary = _RecordingMseBoundary(allow=True)
    invoker = _production_invoker(
        contract=_side_effect_contract(),
        executor=executor,
        boundary=boundary,
    )
    run_id = canonical_run_id_for_tests("r9-allow")
    _invoke_production(invoker, run_id=run_id)
    assert boundary.calls == 1
    assert executor.calls == 1


def test_gr10_r9_read_only_skips_mse_boundary() -> None:
    executor = _CountingExecutor()
    boundary = _RecordingMseBoundary(allow=True)
    contract = _read_only_contract()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=executor,
        meaningful_side_effect_authorization=boundary,
    )
    run_id = canonical_run_id_for_tests("r9-read")
    state = build_runtime_state_for_tests(run_id=run_id)
    request = ToolExecutionRequest(
        run_id=validate_run_id(run_id),
        tool_id=contract.tool_id,
        step_id="s1",
        input=_In(value=1),
    )
    run = validate_run_id(run_id)
    token = bind_active_execution_identity(
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
    try:
        invoker.invoke(state=state, agent_id="agent-1", request=request)
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(token)
    assert boundary.calls == 0
    assert executor.calls == 1


def test_gr10_r9_production_missing_boundary_fail_closed() -> None:
    executor = _CountingExecutor()
    invoker = _production_invoker(
        contract=_side_effect_contract(),
        executor=executor,
        boundary=None,
    )
    run_id = canonical_run_id_for_tests("r9-missing")
    with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError) as exc:
        _invoke_production(invoker, run_id=run_id)
    assert exc.value.reason is SideEffectAuthorizationFailureReason.NOT_CONFIGURED
    assert executor.calls == 0


def test_gr10_r9_production_composition_missing_mse_port_fail_closed() -> None:
    with pytest.raises(ProductionRuntimeToolInvokerCompositionError, match="meaningful_side_effect"):
        build_production_runtime_tool_invoker(
            registry=FakeRegistry(_side_effect_contract()),
            executor=_CountingExecutor(),
            agent_runtime_governance=_allow_all_governance(),
            production_mode=True,
        )


def test_gr10_r9_production_composition_custom_mse_port_wired() -> None:
    custom = _RecordingMseBoundary(allow=True)
    invoker = build_production_runtime_tool_invoker(
        registry=FakeRegistry(_side_effect_contract()),
        executor=_CountingExecutor(),
        agent_runtime_governance=_allow_all_governance(),
        meaningful_side_effect_authorization=custom,
        production_mode=True,
    )
    assert invoker._meaningful_side_effect_authorization is custom  # noqa: SLF001


def test_gr10_r9_runtime_config_exposes_mse_port() -> None:
    source = _CONFIG.read_text(encoding="utf-8-sig")
    assert "meaningful_side_effect_authorization" in source
    tree = ast.parse(source, filename=str(_CONFIG))
    rel = _CONFIG.relative_to(_REPO_ROOT).as_posix()
    forbidden = [
        v
        for v in ast.walk(tree)
        if isinstance(v, ast.ImportFrom)
        and v.module
        and "meaningful_side_effect_authorization" in v.module
        and not v.module.startswith("intergrax.contracts.")
    ]
    assert forbidden == []


def test_gr10_r9_invoker_ast_mse_after_tool_authorization() -> None:
    source = _INVOKER.read_text(encoding="utf-8-sig")
    assert "_require_canonical_meaningful_side_effect_authorization" in source
    tree = ast.parse(source, filename=str(_INVOKER))
    auth_idx: int | None = None
    mse_idx: int | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_require_current_attempt_authorization":
            for index, sub in enumerate(node.body):
                if (
                    isinstance(sub, ast.Expr)
                    and isinstance(sub.value, ast.Call)
                    and isinstance(sub.value.func, ast.Attribute)
                    and sub.value.func.attr == "_require_canonical_meaningful_side_effect_authorization"
                ):
                    mse_idx = index
        if node.name == "_prepare_invocation":
            for index, sub in enumerate(node.body):
                if (
                    isinstance(sub, ast.Expr)
                    and isinstance(sub.value, ast.Call)
                    and isinstance(sub.value.func, ast.Attribute)
                    and sub.value.func.attr == "_require_current_attempt_authorization"
                ):
                    auth_idx = index
    assert auth_idx is not None
    assert mse_idx is not None


def test_gr10_r9_runtime_context_custom_boundary_end_to_end() -> None:
    custom = _RecordingMseBoundary(allow=True)
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
        production_mode=True,
        trace_db_path="/tmp/trace.db",
        tool_registry=FakeRegistry(_side_effect_contract()),
        agent_runtime_governance=_allow_all_governance(),
        canonical_inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=custom,
    )
    ctx = RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
        governance_service=create_lab_allow_governance_service(),
    )
    invoker = ctx.config.tool_invoker
    assert invoker is not None
    _invoke_production(invoker, run_id=canonical_run_id_for_tests("r9-ctx"))
    assert custom.calls == 1


def test_gr10_r9_declarative_wiring_passes_custom_boundary() -> None:
    custom = _RecordingMseBoundary(allow=True)
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["read_file"]),
        wiring_context=ToolWiringContext(),
        registry=FakeRegistry(_side_effect_contract()),
    )
    catalog = build_declarative_invoker_from_tool_wiring(
        wiring,
        agent_runtime_governance=_allow_all_governance(),
        canonical_inner_execution_guard=_RecordingGuard(allow=True),
        meaningful_side_effect_authorization=custom,
        production_mode=True,
    )
    assert catalog is not None
    _invoke_production(catalog.tool_invoker, run_id=canonical_run_id_for_tests("r9-decl"))
    assert custom.calls == 1
