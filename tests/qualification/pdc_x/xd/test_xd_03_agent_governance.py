# © Artur Czarnecki. All rights reserved.

"""XD-03 (Agent <-> Tools/Governance): no permit, no execution; agents own no execution path.

An agent-proposed tool call goes through the canonical production tool invoker
(``build_production_runtime_tool_invoker`` in strict ``production_mode``) with the real governance
pipeline (capability grants, policy engine, audit recorder). The physical tool executor is a counting
handler: the assertions are on whether the executor was ever reached.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.contracts.policy_enforcement_mode import PolicyEnforcementMode
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    materialize_tool_dependency_attempt_boundary,
)
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.core.contracts import ToolContract, ToolRetryPolicy
from intergrax.tools.execution_models import ToolEffectCertainty, ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from pydantic import BaseModel
from testing_support.builder import build_runtime_state_for_tests, canonical_run_id_for_tests
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)
from tests.qualification.pdc_x.d8._d8_support import make_governance
from tests.unit.runtime.governance.test_gr3_r1_explicit_inner_guard_composition import _AllowingTestGuard
from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import _RecordingMseBoundary

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[4]
TENANT = "test-tenant"
AGENT = "agent-1"
TOOL = "xd03.lookup"
OTHER_TOOL = "xd03.delete"


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    result: int


class _CountingExecutor:
    """The physical tool executor: counts how often execution is actually reached."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls.append(request.tool_id)
        return _Out(result=request.input.value)  # type: ignore[attr-defined]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in (TOOL, OTHER_TOOL):
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
            handler=_CountingExecutor(),
        )
    return registry


def _invoker(executor: _CountingExecutor, granted, *, scope: set[str]) -> tuple[RuntimeToolInvoker, object]:  # type: ignore[no-untyped-def]
    harness = make_governance(granted=granted)
    config = tool_dependency_concurrency_admission_configuration(TOOL, max_concurrent_calls=2)
    boundary = materialize_tool_dependency_attempt_boundary(config, production_mode=True)
    assert boundary is not None
    invoker = build_production_runtime_tool_invoker(
        registry=_registry(),
        executor=executor,
        dependency_attempt_boundary=boundary,
        production_mode=True,
        agent_runtime_governance=harness.boundary,
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        inner_execution_guard=_AllowingTestGuard(),
        scope_policy=StaticToolScopePolicy(allowed_tools=scope),
    )
    return invoker, harness


@contextmanager
def _identity(run_id: str, *, tenant: str = TENANT) -> Iterator[None]:
    run = validate_run_id(run_id)
    exec_token = bind_active_execution_identity(
        run_id=run, attempt_id=mint_attempt_id(), execution_id=mint_execution_id()
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(tenant_id=tenant, workspace_id="ws", principal_id="principal-A")
    )
    try:
        yield
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(exec_token)


_LAST: list[object] = []  # the runtime state of the most recent proposal (for trace assertions)


def _propose(invoker: RuntimeToolInvoker, tool_id: str, *, tenant: str = TENANT, agent_id: str = AGENT):  # type: ignore[no-untyped-def]
    """An agent proposes a tool call. Returns ``(result_or_none, error_or_none)``; the state is kept in ``_LAST``."""
    run_id = canonical_run_id_for_tests(f"xd03-{tool_id}-{tenant}")
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="xd03")
    env.policy_rules = PolicyRulesProfile(inline_rules=[], policy_enforcement_mode=PolicyEnforcementMode.ENFORCE)
    state = build_runtime_state_for_tests(run_id=run_id)
    state.context.config.policy_bundle = wire_policy_bundle(env)
    _LAST.clear()
    _LAST.append(state)
    with _identity(run_id, tenant=tenant):
        try:
            result = invoker.invoke(
                state=state,
                agent_id=agent_id,
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id), tool_id=tool_id, step_id="1", input=_In(value=7)
                ),
            )
        except Exception as exc:  # a typed governance denial may surface as an exception
            return None, exc
    return result, None


def _denied(result: ToolExecutionResult[BaseModel] | None, error: Exception | None) -> bool:
    if error is not None:
        return True
    assert result is not None
    return result.success is False


def test_xd03_a_permitted_tool_call_reaches_the_executor_and_is_audited() -> None:
    executor = _CountingExecutor()
    invoker, harness = _invoker(executor, {(TENANT, AGENT): frozenset({TOOL})}, scope={TOOL})
    result, error = _propose(invoker, TOOL)
    invoker.close()  # type: ignore[attr-defined]
    assert error is None and result is not None and result.success is True
    assert executor.calls == [TOOL]
    allowed = [e for e in harness.sink.events if e.decision is ToolAuthorizationDecisionState.ALLOW]  # type: ignore[attr-defined]
    assert allowed and allowed[0].agent_id == AGENT and allowed[0].tool_id == TOOL


def test_xd03_a_tool_call_without_a_governance_permit_is_denied_and_never_executed() -> None:
    executor = _CountingExecutor()
    invoker, harness = _invoker(executor, {}, scope={TOOL})  # the agent holds no capability grant
    result, error = _propose(invoker, TOOL)
    invoker.close()  # type: ignore[attr-defined]
    assert _denied(result, error)
    if result is not None:
        assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED
    assert executor.calls == [], "the physical executor was never reached"
    # The refusal is recorded as canonical trace evidence on the run (the capability denial is raised
    # before the governance audit recorder runs, so the audit sink stays empty: see the finding below).
    steps = [event.step for event in _LAST[0].trace_events]  # type: ignore[attr-defined]
    assert "agent_runtime_governance_capability_not_granted" in steps


def test_xd03_a_missing_permit_denial_reaches_the_governance_audit_sink() -> None:
    executor = _CountingExecutor()
    invoker, harness = _invoker(executor, {}, scope={TOOL})
    result, error = _propose(invoker, TOOL)
    invoker.close()  # type: ignore[attr-defined]
    assert _denied(result, error)
    denied = [e for e in harness.sink.events if e.decision is ToolAuthorizationDecisionState.DENY]  # type: ignore[attr-defined]
    assert denied and denied[0].tool_id == TOOL


def test_xd03_a_grant_for_one_tool_does_not_permit_another() -> None:
    executor = _CountingExecutor()
    invoker, _ = _invoker(executor, {(TENANT, AGENT): frozenset({TOOL})}, scope={TOOL, OTHER_TOOL})
    other_result, other_error = _propose(invoker, OTHER_TOOL)
    invoker.close()  # type: ignore[attr-defined]
    assert _denied(other_result, other_error)
    assert executor.calls == []


def test_xd03_another_tenants_grant_does_not_permit_the_call() -> None:
    executor = _CountingExecutor()
    invoker, _ = _invoker(executor, {("tenant-other", AGENT): frozenset({TOOL})}, scope={TOOL})
    result, error = _propose(invoker, TOOL)
    invoker.close()  # type: ignore[attr-defined]
    assert _denied(result, error)
    assert executor.calls == []


def test_xd03_another_agents_grant_does_not_permit_the_call() -> None:
    executor = _CountingExecutor()
    invoker, _ = _invoker(executor, {(TENANT, "agent-2"): frozenset({TOOL})}, scope={TOOL})
    result, error = _propose(invoker, TOOL, agent_id=AGENT)
    invoker.close()  # type: ignore[attr-defined]
    assert _denied(result, error)
    assert executor.calls == []


def test_xd03_a_granted_tool_outside_the_scope_policy_is_still_denied() -> None:
    executor = _CountingExecutor()
    invoker, _ = _invoker(executor, {(TENANT, AGENT): frozenset({TOOL})}, scope={OTHER_TOOL})
    result, error = _propose(invoker, TOOL)
    invoker.close()  # type: ignore[attr-defined]
    assert _denied(result, error)
    assert executor.calls == []


def test_xd03_strict_composition_refuses_to_build_without_governance() -> None:
    from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
        ProductionRuntimeToolInvokerCompositionError,
    )

    config = tool_dependency_concurrency_admission_configuration(TOOL, max_concurrent_calls=2)
    boundary = materialize_tool_dependency_attempt_boundary(config, production_mode=True)
    with pytest.raises(ProductionRuntimeToolInvokerCompositionError):
        build_production_runtime_tool_invoker(
            registry=_registry(),
            executor=_CountingExecutor(),
            dependency_attempt_boundary=boundary,
            production_mode=True,
            agent_runtime_governance=None,
            meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
            inner_execution_guard=_AllowingTestGuard(),
            scope_policy=StaticToolScopePolicy(allowed_tools={TOOL}),
        )


def _python_files(*roots: str) -> list[Path]:
    return [p for root in roots for p in sorted((REPO_ROOT / root).rglob("*.py")) if "__pycache__" not in p.parts]


def test_xd03_no_agent_or_application_module_owns_a_tool_execution_path() -> None:
    """Tier-2 agents and Tier-3 applications never import or implement the physical ``ToolExecutor``."""
    offenders: list[str] = []
    for path in _python_files("agents", "applications"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "ToolExecutor" not in text and "tool_executor" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            imported = (
                [alias.name for alias in node.names] + [getattr(node, "module", "") or ""]
                if isinstance(node, ast.ImportFrom | ast.Import)
                else []
            )
            defined = [node.name] if isinstance(node, ast.ClassDef) else []
            if any("ToolExecutor" in name or "tool_executor" in name for name in [*imported, *defined]):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
                break
    assert offenders == []
