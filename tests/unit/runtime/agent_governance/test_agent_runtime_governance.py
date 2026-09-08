# © Artur Czarnecki. All rights reserved.

"""NPSC-4 agent runtime governance unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    ApprovalRequestStatus,
    CapabilityGrant,
    ToolAuthorizationDecisionState,
    ToolAuthorizationRequest,
    ToolAuthorizationRiskLevel,
)
from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.runtime.agent_governance.approval_boundary import (
    AgentRuntimeApprovalBoundary,
    InMemoryApprovalStore,
)
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
    CapabilityNotGrantedError,
    ToolGovernanceApprovalRequiredError,
    ToolGovernanceDeniedError,
)
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    AllowAllPolicyProvider,
    DenyCapabilityPolicyProvider,
    FinancialApprovalPolicyProvider,
    HighRiskApprovalPolicyProvider,
)

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_01234567890123456789012345678901")
_RUN_ID = RunId("run_01234567890123456789012345678901")
_ATTEMPT_ID = AttemptId("attempt_01234567890123456789012345678901")
_AGENT = AgentIdentity(agent_id="invoice-agent", tenant_id="tenant-a")


def _request(
    *,
    capability: str = "read_invoice",
    tool_id: str = "invoice.read",
    risk: ToolAuthorizationRiskLevel = ToolAuthorizationRiskLevel.LOW,
    approval_evidence_ref: str | None = None,
) -> ToolAuthorizationRequest:
    return ToolAuthorizationRequest(
        agent=_AGENT,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        capability=capability,
        tool_id=tool_id,
        requested_action=f"execute:{tool_id}",
        risk_level=risk,
        approval_evidence_ref=approval_evidence_ref,
    )


def _pipeline(
    *,
    grants: tuple[CapabilityGrant, ...] = (),
    policies: tuple = (),
    with_approval: bool = True,
) -> AgentRuntimeGovernancePipeline:
    sink = InMemoryGovernanceAuditSink()
    recorder = GovernanceAuditRecorder(sink)
    store = InMemoryApprovalStore()
    approval = AgentRuntimeApprovalBoundary(store) if with_approval else None
    return AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver(grants),
        policy_engine=AgentRuntimePolicyEngine(policies),
        audit_recorder=recorder,
        approval_boundary=approval,
    )


def test_capability_evaluation_allows_granted_capability() -> None:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"read_invoice", "search_customer"}),
    )
    pipeline = _pipeline(grants=(grant,), policies=(AllowAllPolicyProvider(),))
    decision = pipeline.evaluate(_request())
    assert decision.is_allowed


def test_capability_evaluation_denies_missing_grant() -> None:
    pipeline = _pipeline(policies=(AllowAllPolicyProvider(),))
    with pytest.raises(CapabilityNotGrantedError):
        pipeline.evaluate(_request())


def test_capability_evaluation_denies_explicitly_denied() -> None:
    denied_grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"read_invoice"}),
        denied_capabilities=frozenset({"approve_payment"}),
    )
    pipeline = _pipeline(grants=(denied_grant,), policies=(AllowAllPolicyProvider(),))
    with pytest.raises(CapabilityNotGrantedError):
        pipeline.evaluate(_request(capability="approve_payment"))


def test_policy_evaluation_deny_wins_over_allow() -> None:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"approve_payment"}),
    )
    pipeline = _pipeline(
        grants=(grant,),
        policies=(
            AllowAllPolicyProvider(),
            DenyCapabilityPolicyProvider(denied_capabilities=frozenset({"approve_payment"})),
        ),
    )
    decision = pipeline.evaluate(_request(capability="approve_payment"))
    assert decision.decision is ToolAuthorizationDecisionState.DENY


def test_financial_policy_requires_approval() -> None:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"approve_payment"}),
    )
    pipeline = _pipeline(
        grants=(grant,),
        policies=(FinancialApprovalPolicyProvider(),),
    )
    decision = pipeline.evaluate(_request(capability="approve_payment"))
    assert decision.requires_approval
    assert "approval_id=" in decision.reason


def test_financial_policy_allows_with_evidence() -> None:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"approve_payment"}),
    )
    pipeline = _pipeline(
        grants=(grant,),
        policies=(FinancialApprovalPolicyProvider(),),
    )
    decision = pipeline.evaluate(
        _request(capability="approve_payment", approval_evidence_ref="approval-xyz"),
    )
    assert decision.is_allowed


def test_high_risk_policy_requires_approval() -> None:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"read_invoice"}),
    )
    pipeline = _pipeline(
        grants=(grant,),
        policies=(HighRiskApprovalPolicyProvider(),),
    )
    decision = pipeline.evaluate(_request(risk=ToolAuthorizationRiskLevel.HIGH))
    assert decision.requires_approval


def test_authorization_boundary_raises_on_deny() -> None:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"approve_payment"}),
    )
    boundary = AgentRuntimeGovernanceBoundary(
        _pipeline(
            grants=(grant,),
            policies=(
                DenyCapabilityPolicyProvider(denied_capabilities=frozenset({"approve_payment"})),
            ),
        ),
    )
    with pytest.raises(ToolGovernanceDeniedError):
        boundary.authorize_tool(_request(capability="approve_payment"))


def test_authorization_boundary_raises_on_approval_required() -> None:
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"approve_payment"}),
    )
    boundary = AgentRuntimeGovernanceBoundary(
        _pipeline(grants=(grant,), policies=(FinancialApprovalPolicyProvider(),)),
    )
    with pytest.raises(ToolGovernanceApprovalRequiredError) as exc_info:
        boundary.authorize_tool(_request(capability="approve_payment"))
    assert exc_info.value.approval_id.startswith("approval_")


def test_approval_lifecycle() -> None:
    store = InMemoryApprovalStore()
    boundary = AgentRuntimeApprovalBoundary(store)
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    approval = boundary.create_approval_request(
        _request(capability="approve_payment", risk=ToolAuthorizationRiskLevel.HIGH),
        expires_at=expires,
    )
    assert approval.status is ApprovalRequestStatus.WAITING_FOR_APPROVAL

    approved = boundary.approve(approval.approval_id, decided_by="user-1")
    assert approved.status is ApprovalRequestStatus.APPROVED
    assert boundary.is_approved(approval.approval_id)


def test_approval_rejection() -> None:
    store = InMemoryApprovalStore()
    boundary = AgentRuntimeApprovalBoundary(store)
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    approval = boundary.create_approval_request(
        _request(risk=ToolAuthorizationRiskLevel.HIGH),
        expires_at=expires,
    )
    rejected = boundary.reject(approval.approval_id, decided_by="user-1")
    assert rejected.status is ApprovalRequestStatus.REJECTED
    assert not boundary.is_approved(approval.approval_id)


def test_approval_expiration() -> None:
    store = InMemoryApprovalStore()
    boundary = AgentRuntimeApprovalBoundary(store)
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    approval = boundary.create_approval_request(
        _request(risk=ToolAuthorizationRiskLevel.HIGH),
        expires_at=past,
    )
    expired = boundary.check_expiration(approval.approval_id)
    assert expired.status is ApprovalRequestStatus.EXPIRED


def test_audit_events_generated_on_decision() -> None:
    sink = InMemoryGovernanceAuditSink()
    recorder = GovernanceAuditRecorder(sink)
    grant = CapabilityGrant(
        agent_id="invoice-agent",
        tenant_id="tenant-a",
        allowed_capabilities=frozenset({"read_invoice"}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=recorder,
        approval_boundary=None,
    )
    pipeline.evaluate(_request())
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.agent_id == "invoice-agent"
    assert event.tool_id == "invoice.read"
    assert event.decision is ToolAuthorizationDecisionState.ALLOW
    assert event.event_id.startswith("governance_evt_")
