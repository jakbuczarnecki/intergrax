# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.agent_governance_hitl import digest_logical_invocation_fingerprint
from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    ToolAuthorizationRequest,
    ToolAuthorizationRiskLevel,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.entity_id import (
    mint_suspended_operation_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.suspended_operation.in_memory_store import (
    InMemorySuspendedExecutionOperationStore,
)
from intergrax.runtime.nexus.tools.agent_governance_approval_pause_bridge import (
    translate_agent_governance_approval_error,
)
from intergrax.runtime.agent_governance.errors import ToolGovernanceApprovalRequiredError
from tests.unit.runtime.execution.suspended_operation.test_suspended_operation_store import (
    _descriptor,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[5]
BRIDGE = (
    REPO
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "agent_governance_approval_pause_bridge.py"
)


def test_signal_idempotency_ignores_approval_evidence_ref() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    auth = ToolAuthorizationRequest(
        agent=AgentIdentity(agent_id="agent-a", tenant_id="tenant-a"),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        capability="sandbox",
        tool_id="code.exec",
        requested_action="execute:code.exec",
        risk_level=ToolAuthorizationRiskLevel.HIGH,
        approval_evidence_ref="audit-B",
    )
    error = ToolGovernanceApprovalRequiredError(
        run_id=str(run_id),
        agent_id="agent-a",
        tool_id="code.exec",
        capability="sandbox",
        approval_id="agr-1",
        reason="needs approval",
        policy_results=(),
    )
    signal = translate_agent_governance_approval_error(
        error,
        authorization_request=auth,
        execution_id=str(auth.execution_id),
        step_id="step-1",
        idempotency_key="idem-A",
    )
    assert signal.idempotency_key == "idem-A"
    digest = digest_logical_invocation_fingerprint(
        task_id=str(signal.task_id),
        run_id=str(signal.run_id),
        attempt_id=str(signal.attempt_id),
        execution_id=str(signal.execution_id),
        tenant_id=signal.tenant_id,
        agent_id=signal.agent_id,
        tool_id=signal.tool_id,
        step_id=signal.step_id,
        idempotency_key=signal.idempotency_key,
        invocation_intent_digest="sha256:" + ("a" * 64),
    )
    assert "idem-A" in digest.digest or digest.digest


def test_signal_idempotency_none_when_not_provided() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    auth = ToolAuthorizationRequest(
        agent=AgentIdentity(agent_id="agent-a", tenant_id="tenant-a"),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        capability="sandbox",
        tool_id="code.exec",
        requested_action="execute:code.exec",
        risk_level=ToolAuthorizationRiskLevel.HIGH,
    )
    error = ToolGovernanceApprovalRequiredError(
        run_id=str(run_id),
        agent_id="agent-a",
        tool_id="code.exec",
        capability="sandbox",
        approval_id="agr-1",
        reason="needs approval",
        policy_results=(),
    )
    signal = translate_agent_governance_approval_error(
        error,
        authorization_request=auth,
        execution_id=str(auth.execution_id),
        step_id="step-1",
        idempotency_key=None,
    )
    assert signal.idempotency_key is None


def test_store_enforces_single_active_descriptor_per_fingerprint() -> None:
    store = InMemorySuspendedExecutionOperationStore()
    first = _descriptor()
    fingerprint = digest_logical_invocation_fingerprint(
        task_id=str(first.identity.task_id),
        run_id=str(first.identity.run_id),
        attempt_id=str(first.identity.attempt_id),
        execution_id=str(first.identity.execution_id),
        tenant_id="tenant-a",
        agent_id="agent-a",
        tool_id="code.exec",
        step_id="step-1",
        idempotency_key="idem-1",
        invocation_intent_digest="sha256:" + ("b" * 64),
    )
    first = first.model_copy(
        update={"logical_invocation_fingerprint": fingerprint},
    )
    applied = store.prepare(first)
    assert applied.outcome is SuspendedOperationMutationOutcome.APPLIED
    second = SuspendedExecutionOperationDescriptor(
        suspended_operation_id=mint_suspended_operation_id(),
        operation_kind=first.operation_kind,
        identity=first.identity,
        continuation_id="gcr_other",
        invocation_scope_id="agr_other",
        materialization_state=SuspendedOperationMaterializationState.PREPARED,
        materialization_revision=0,
        payload_digest=first.payload_digest,
        payload=first.payload,
        logical_invocation_fingerprint=fingerprint,
    )
    duplicate = store.prepare(second)
    assert duplicate.outcome is SuspendedOperationMutationOutcome.ALREADY_ACTIVE
    assert duplicate.descriptor is not None
    assert duplicate.descriptor.suspended_operation_id == first.suspended_operation_id


def test_bridge_gate_bans_approval_evidence_ref_as_idempotency_source() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    assert "idempotency_key=authorization_request.approval_evidence_ref" not in source
