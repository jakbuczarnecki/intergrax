# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

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
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.agent_governance.grant_verifier import (
    AgentGovernanceGrantVerificationError,
    AgentGovernanceGrantVerifier,
)
from intergrax.runtime.execution.suspended_operation.store_engine import (
    SuspendedOperationBackingStore,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_LIFECYCLE_ADAPTER = (
    _REPO / "intergrax/runtime/human/agent_governance_grant_lifecycle.py"
)
_STORE_ENGINE = (
    _REPO / "intergrax/runtime/execution/suspended_operation/store_engine.py"
)
_AUTH_SCOPE_COMPAT = (
    _REPO
    / "intergrax/contracts/execution/suspended_operation/authority_scope_compat.py"
)
_DESCRIPTOR = _REPO / "intergrax/contracts/execution/suspended_operation/descriptor.py"


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
        payload_digest="sha256:" + ("a" * 64),
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
        policy_provenance_digest="sha256:" + ("b" * 64),
    )


def _pending(
    requirement: AgentGovernanceHumanApprovalRequirement,
) -> AgentGovernanceHumanApprovalPending:
    return AgentGovernanceHumanApprovalPending(
        agent_governance_invocation_scope_id=requirement.agent_governance_invocation_scope_id,
        requirement=requirement,
        task_id=requirement.task_id,
        run_id=requirement.run_id,
        attempt_id=requirement.attempt_id,
        execution_id=requirement.execution_id,
        tenant_id=requirement.tenant_id,
        agent_id=requirement.agent_id,
        tool_id=requirement.tool_id,
        step_id=requirement.step_id,
        idempotency_key=requirement.idempotency_key,
        human_request_id="hr_test",
        pause_id="pause_test",
        policy_provenance_digest=requirement.policy_provenance_digest,
        created_at="2026-09-23T00:00:00+00:00",
        generation=1,
    )


def _grant(
    requirement: AgentGovernanceHumanApprovalRequirement,
    pending: AgentGovernanceHumanApprovalPending,
) -> AgentGovernanceHumanApprovalGrant:
    return AgentGovernanceHumanApprovalGrant(
        grant_id="grant_test",
        agent_governance_invocation_scope_id=requirement.agent_governance_invocation_scope_id,
        pending_generation=1,
        logical_invocation_fingerprint=requirement.logical_invocation_fingerprint,
        task_id=requirement.task_id,
        run_id=requirement.run_id,
        attempt_id=requirement.attempt_id,
        execution_id=requirement.execution_id,
        tenant_id=requirement.tenant_id,
        agent_id=requirement.agent_id,
        tool_id=requirement.tool_id,
        step_id=requirement.step_id,
        idempotency_key=requirement.idempotency_key,
        policy_provenance_digest=requirement.policy_provenance_digest,
        human_request_id=pending.human_request_id,
        pause_id=pending.pause_id,
        approved_at="2026-09-23T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
    )


def test_verifier_requires_pending_for_available_grant() -> None:
    requirement = _requirement()
    grant = _grant(requirement, _pending(requirement))
    record = AgentGovernanceGrantLifecycleRecord(
        grant=grant,
        lifecycle_state=AgentGovernanceGrantLifecycleState.AVAILABLE,
        lifecycle_revision=1,
    )
    verifier = AgentGovernanceGrantVerifier()
    with pytest.raises(AgentGovernanceGrantVerificationError, match="pending required"):
        verifier.verify_for_resume(
            lifecycle_record=record,
            pending=None,
            requirement=requirement,
            request=requirement.authorization_request,
            logical_invocation_fingerprint=requirement.logical_invocation_fingerprint,
            pause_generation=1,
        )


def test_verifier_rejects_policy_digest_mismatch() -> None:
    requirement = _requirement()
    pending = _pending(requirement)
    grant = _grant(requirement, pending)
    grant = grant.model_copy(
        update={"policy_provenance_digest": "sha256:" + ("c" * 64)}
    )
    record = AgentGovernanceGrantLifecycleRecord(
        grant=grant,
        lifecycle_state=AgentGovernanceGrantLifecycleState.AVAILABLE,
        lifecycle_revision=1,
    )
    verifier = AgentGovernanceGrantVerifier()
    with pytest.raises(AgentGovernanceGrantVerificationError, match="policy digest"):
        verifier.verify_for_resume(
            lifecycle_record=record,
            pending=pending,
            requirement=requirement,
            request=requirement.authorization_request,
            logical_invocation_fingerprint=requirement.logical_invocation_fingerprint,
            pause_generation=1,
        )


def test_lifecycle_adapter_uses_checkpoint_persistence_contract() -> None:
    source = _LIFECYCLE_ADAPTER.read_text(encoding="utf-8")
    assert "TaskCheckpointPersistence" in source
    assert "StaleCheckpointWriteError" in source
    assert "type: ignore" not in source
    assert (
        "self._task.runtime.governance.agent_governance_human_approval_grant ="
        not in source
    )
    assert "self._checkpoint_store.save" in source
    assert "_canonical_checkpoint_revision" not in source
    assert "expected_checkpoint_revision=snapshot.checkpoint_revision" in source


def test_authority_scope_unknown_cannot_map_to_mse() -> None:
    source = _AUTH_SCOPE_COMPAT.read_text(encoding="utf-8")
    assert "UnknownInvocationScopeError" in source
    assert "is_meaningful_side_effect_invocation_scope_id" in source
    assert "raise UnknownInvocationScopeError" in source


def test_descriptor_v1_dhr_without_authority_scope_remains_readable() -> None:
    source = _DESCRIPTOR.read_text(encoding="utf-8")
    assert "_migrate_legacy_authority_scope" in source
    assert "_default_authority_scope" not in source
    assert "recognize_authority_scope_from_invocation" in source


def test_reblock_and_block_share_continuation_helpers() -> None:
    source = _STORE_ENGINE.read_text(encoding="utf-8")
    assert source.count("assert_execution_continuation_identity_match") >= 2
    assert source.count("assert_governed_correlation_matches_continuation") >= 2
    assert "_ = next_authority_scope" not in source
    assert "authority_scope" in source


def test_store_engine_has_no_ignored_authority_scope() -> None:
    assert (
        "_ = next_authority_scope"
        not in SuspendedOperationBackingStore.authority_reblock_from_claimed.__code__.co_consts
    )
