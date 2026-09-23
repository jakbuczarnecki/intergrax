# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.1 — Agent Governance typed approval contracts."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleRecord,
    AgentGovernanceGrantLifecycleState,
    AgentGovernanceHumanApprovalGrant,
    AgentGovernanceHumanApprovalPending,
    AgentGovernanceHumanApprovalRequirement,
    LogicalInvocationFingerprint,
    digest_logical_invocation_fingerprint,
    mint_agent_governance_invocation_scope_id,
)
from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    ToolAuthorizationRequest,
    ToolAuthorizationRiskLevel,
)
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.runtime.agent_governance.grant_verifier import (
    AgentGovernanceGrantVerificationError,
    AgentGovernanceGrantVerifier,
)

pytestmark = pytest.mark.unit


def _fingerprint() -> LogicalInvocationFingerprint:
    return digest_logical_invocation_fingerprint(
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        attempt_id=str(mint_attempt_id()),
        execution_id=str(mint_execution_id()),
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_id="tool-a",
        step_id="step-a",
        idempotency_key="idem",
        payload_digest="sha256:" + "a" * 64,
    )


def _requirement() -> AgentGovernanceHumanApprovalRequirement:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    scope = mint_agent_governance_invocation_scope_id()
    auth = ToolAuthorizationRequest(
        agent=AgentIdentity(agent_id="agent-a", tenant_id="tenant-a"),
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        capability="approve_payment",
        tool_id="tool-a",
        requested_action="execute:tool-a",
        risk_level=ToolAuthorizationRiskLevel.HIGH,
    )
    return AgentGovernanceHumanApprovalRequirement(
        agent_governance_invocation_scope_id=scope,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_id="tool-a",
        step_id="step-a",
        idempotency_key="idem",
        approval_id="approval_test",
        authorization_request=auth,
        logical_invocation_fingerprint=_fingerprint(),
        pause_generation=1,
    )


def test_contracts_are_frozen_and_authority_scoped() -> None:
    requirement = _requirement()
    pending = AgentGovernanceHumanApprovalPending(
        agent_governance_invocation_scope_id=requirement.agent_governance_invocation_scope_id,
        requirement=requirement,
        task_id=requirement.task_id,
        run_id=requirement.run_id,
        attempt_id=requirement.attempt_id,
        execution_id=requirement.execution_id,
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_id="tool-a",
        step_id="step-a",
        idempotency_key="idem",
        human_request_id="hr_test",
        pause_id="pause_test",
        created_at="2026-09-23T00:00:00+00:00",
        generation=1,
    )
    grant = AgentGovernanceHumanApprovalGrant(
        grant_id="grant_test",
        agent_governance_invocation_scope_id=requirement.agent_governance_invocation_scope_id,
        pending_generation=1,
        logical_invocation_fingerprint=requirement.logical_invocation_fingerprint,
        task_id=requirement.task_id,
        run_id=requirement.run_id,
        attempt_id=requirement.attempt_id,
        execution_id=requirement.execution_id,
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_id="tool-a",
        step_id="step-a",
        idempotency_key="idem",
        human_request_id="hr_test",
        pause_id="pause_test",
        approved_at="2026-09-23T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    record = AgentGovernanceGrantLifecycleRecord(
        grant=grant,
        lifecycle_state=AgentGovernanceGrantLifecycleState.AVAILABLE,
        lifecycle_revision=1,
    )
    assert record.grant.grant_id == "grant_test"
    assert pending.requirement.approval_id == "approval_test"


def test_verifier_rejects_cross_authority_artifacts() -> None:
    verifier = AgentGovernanceGrantVerifier()
    with pytest.raises(AgentGovernanceGrantVerificationError):
        verifier.reject_foreign_approval_artifact("arbitrary")
    with pytest.raises(AgentGovernanceGrantVerificationError):
        verifier.reject_foreign_approval_artifact(
            DeclarativeHitlApprovalGrant(
                grant_id="g",
                invocation_scope_id="dhr_x",
                task_id=str(mint_task_id()),
                run_id=str(mint_run_id()),
                step_id="s",
                tool_id="t",
                agent_id="a",
                idempotency_key=None,
                matched_rule_ids=(),
                human_request_id="hr",
                policy_provenance_digest=None,
                pause_id="p",
                approved_at="2026-09-23T00:00:00+00:00",
            )
        )
    with pytest.raises(AgentGovernanceGrantVerificationError):
        verifier.reject_foreign_approval_artifact(
            GovernedContinuationApprovalGrant(
                grant_id="g",
                continuation_request_id="c",
                side_effect_scope_id="mse",
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=mint_execution_id(),
                operation_id="op",
                pause_id="p",
                human_request_id="hr",
                approved_at="2026-09-23T00:00:00+00:00",
            )
        )
