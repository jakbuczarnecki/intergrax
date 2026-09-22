# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5-R4 — execution-bound governance approval evidence contract."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    build_execution_bound_catalog_tool_invoker_for_qualified_capability,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.tool_invocation_governance_approval_evidence import (
    ToolInvocationGovernanceApprovalEvidence,
)
from intergrax.runtime.agent_governance.request_builder import (
    build_tool_authorization_request,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_high_risk_tool_approval_evidence,
    uca6c_strict_sandbox_env_profile,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from intergrax.tools.registry import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _AllowMse:
    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        _ = request, source_agent_id, source_step_id
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test",
            policy_rule_id="test.mse",
        )
        enforcement = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=True,
            decision=decision,
            enforcement_result=enforcement,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )


def _strict_catalog_invoker() -> NexusExecutionBoundCatalogToolInvoker:
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled_bundles=["sandbox"]),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    invoker = build_execution_bound_catalog_tool_invoker_for_qualified_capability(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=_AllowMse(),
    )
    assert isinstance(invoker, NexusExecutionBoundCatalogToolInvoker)
    return invoker


def test_public_invoke_request_exposes_governance_approval_evidence_field() -> None:
    fields = {
        f.name for f in dataclasses.fields(ExecutionBoundCatalogToolInvokeRequest)
    }
    assert "governance_approval_evidence" in fields


@dataclass
class _EvidenceRecordingCatalogInvoker:
    caller_agent_id: str = "worker-uca6c-qualified"
    last_evidence: ToolInvocationGovernanceApprovalEvidence | None = None

    def invoke(
        self,
        request: ExecutionBoundCatalogToolInvokeRequest,
    ) -> ToolExecutionResult[BaseModel]:
        self.last_evidence = request.governance_approval_evidence
        return ToolExecutionResult.fail("tool_error", "stub")


def test_custom_catalog_invoker_receives_typed_approval_evidence() -> None:
    recording = _EvidenceRecordingCatalogInvoker()
    run_id = str(mint_run_id())
    step_id = "uca6c.bound:exec-1"
    evidence = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id=step_id,
    )
    recording.invoke(
        ExecutionBoundCatalogToolInvokeRequest(
            tool_id=CODE_EXEC_TOOL_ID,
            input=CodeExecInput(code="1", language="python", timeout_s=5),
            tenant_id=_TENANT,
            task_id=str(_TASK_ID),
            run_id=run_id,
            agent_id=recording.caller_agent_id,
            step_id=step_id,
            governance_approval_evidence=evidence,
        ),
    )
    assert recording.last_evidence is evidence
    assert recording.last_evidence.evidence_ref.startswith("uca6c-hitl-grant:")


def test_nexus_maps_request_evidence_to_runtime_state_grant() -> None:
    catalog = _strict_catalog_invoker()
    run_id = str(mint_run_id())
    step_id = "uca6c.bound:nexus-evidence"
    evidence = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id=step_id,
    )
    state = catalog._runtime_state(
        ExecutionBoundCatalogToolInvokeRequest(
            tool_id=CODE_EXEC_TOOL_ID,
            input=CodeExecInput(code="1", language="python", timeout_s=5),
            tenant_id=_TENANT,
            task_id=str(_TASK_ID),
            run_id=run_id,
            agent_id="worker-uca6c-qualified",
            step_id=step_id,
            governance_approval_evidence=evidence,
        ),
    )
    assert state.declarative_hitl_grant is not None
    assert state.declarative_hitl_grant.grant_id == evidence.evidence_ref


def test_governance_request_approval_ref_from_public_evidence() -> None:
    from intergrax.tools.providers.sandbox.bundle import sandbox_exec_contract

    run_id = str(mint_run_id())
    step_id = "uca6c.bound:gov-ref"
    evidence = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id=step_id,
    )
    grant_id = evidence.evidence_ref
    state = RuntimeState(
        context=object(),  # type: ignore[arg-type]
        request=RuntimeRequest(
            agent_id="worker-uca6c-qualified",
            user_id="u",
            session_id=run_id,
            tenant_id=_TENANT,
            message="test",
            task_id=str(_TASK_ID),
            run_id=run_id,
        ),
        run_id=run_id,
        declarative_hitl_grant=None,
    )
    from intergrax.runtime.nexus.tools.governance_approval_evidence_adapter import (
        declarative_hitl_grant_from_invocation_evidence,
    )

    state.declarative_hitl_grant = declarative_hitl_grant_from_invocation_evidence(
        evidence,
    )
    contract = sandbox_exec_contract()
    tool_request = ToolExecutionRequest(
        run_id=run_id,
        step_id=step_id,
        tool_id=CODE_EXEC_TOOL_ID,
        input=CodeExecInput(code="1", language="python", timeout_s=5),
    )
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        auth = build_tool_authorization_request(
            state=state,
            agent_id="worker-uca6c-qualified",
            contract=contract,
            request=tool_request,
        )
    finally:
        reset_active_execution_identity(token)
    assert auth.approval_evidence_ref == grant_id


def test_wrong_scope_evidence_fails_closed() -> None:
    catalog = _strict_catalog_invoker()
    run_id = str(mint_run_id())
    evidence = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id="uca6c.bound:expected",
    )
    with pytest.raises(ValueError, match="step_id mismatch"):
        catalog._runtime_state(
            ExecutionBoundCatalogToolInvokeRequest(
                tool_id=CODE_EXEC_TOOL_ID,
                input=CodeExecInput(code="1", language="python", timeout_s=5),
                tenant_id=_TENANT,
                task_id=str(_TASK_ID),
                run_id=run_id,
                agent_id="worker-uca6c-qualified",
                step_id="uca6c.bound:wrong",
                governance_approval_evidence=evidence,
            ),
        )


def test_repeated_invoke_without_evidence_does_not_reuse_prior_grant() -> None:
    catalog = _strict_catalog_invoker()
    run_id = str(mint_run_id())
    step_id = "uca6c.bound:stale-check"
    evidence = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id=step_id,
    )
    base = ExecutionBoundCatalogToolInvokeRequest(
        tool_id=CODE_EXEC_TOOL_ID,
        input=CodeExecInput(code="1", language="python", timeout_s=5),
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        agent_id="worker-uca6c-qualified",
        step_id=step_id,
    )
    with_grant = dataclasses.replace(base, governance_approval_evidence=evidence)
    state_with = catalog._runtime_state(with_grant)
    state_without = catalog._runtime_state(base)
    assert state_with.declarative_hitl_grant is not None
    assert state_without.declarative_hitl_grant is None


def test_concurrent_evidence_requests_are_isolated() -> None:
    catalog = _strict_catalog_invoker()
    run_id = str(mint_run_id())
    step_a = "uca6c.bound:a"
    step_b = "uca6c.bound:b"
    evidence_a = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id=step_a,
    )
    evidence_b = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id=step_b,
    )

    def _state(step_id: str, evidence: ToolInvocationGovernanceApprovalEvidence):
        return catalog._runtime_state(
            ExecutionBoundCatalogToolInvokeRequest(
                tool_id=CODE_EXEC_TOOL_ID,
                input=CodeExecInput(code="1", language="python", timeout_s=5),
                tenant_id=_TENANT,
                task_id=str(_TASK_ID),
                run_id=run_id,
                agent_id="worker-uca6c-qualified",
                step_id=step_id,
                governance_approval_evidence=evidence,
            ),
        )

    grant_a = _state(step_a, evidence_a).declarative_hitl_grant
    grant_b = _state(step_b, evidence_b).declarative_hitl_grant
    assert grant_a is not None and grant_b is not None
    assert grant_a.grant_id != grant_b.grant_id
    assert grant_a.step_id == step_a
    assert grant_b.step_id == step_b
