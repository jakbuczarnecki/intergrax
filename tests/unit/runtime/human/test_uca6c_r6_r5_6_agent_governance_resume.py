# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.6 — Agent Governance grant + resume wiring."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_governance_grant_lifecycle_port import (
    AgentGovernanceGrantLifecycleOutcome,
)
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleState,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.agent_governance_human_approval_grant import (
    AgentGovernanceHumanApprovalGrantCoordinator,
    AgentGovernanceHumanApprovalGrantError,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import HumanApprovalResolution
from tests.unit.runtime.human.test_agent_governance_grant_lifecycle_h1 import (
    _MemoryTaskCheckpointStore,
    _grant,
    _seed_checkpoint,
    _task,
)
from tests.unit.runtime.architecture.test_uca_6c_r6_r5_foundation_hardening_gates import (
    _pending,
    _requirement,
)

pytestmark = pytest.mark.unit


def _attach_resolution(task, pending) -> None:
    task.runtime.governance.agent_governance_hitl_pending = pending
    task.task_id = pending.task_id
    task.runtime.governance.hitl_resolution = HumanApprovalResolution(
        task_id=pending.task_id,
        pause_id=pending.pause_id,
        human_request_id=pending.human_request_id,
        verdict=HumanResponseVerdict.APPROVE,
        approver=local_development_approver_evidence(tenant_id=task.tenant_id),
        resolved_at="2026-09-23T12:00:00+00:00",
    )


def test_approve_materializes_available_grant_once() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    requirement = _requirement()
    pending = _pending(requirement)
    _attach_resolution(task, pending)
    approver = local_development_approver_evidence(tenant_id=task.tenant_id)

    first = AgentGovernanceHumanApprovalGrantCoordinator.persist_available_grant_from_human_approve(
        task,
        checkpoint_store=store,
        approver=approver,
    )
    assert first.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED
    record = task.runtime.governance.agent_governance_human_approval_grant
    assert record is not None
    assert record.lifecycle_state is AgentGovernanceGrantLifecycleState.AVAILABLE

    second = AgentGovernanceHumanApprovalGrantCoordinator.persist_available_grant_from_human_approve(
        task,
        checkpoint_store=store,
        approver=approver,
    )
    assert second.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED
    assert (
        task.runtime.governance.agent_governance_human_approval_grant.grant.grant_id
        == record.grant.grant_id
    )


def test_reject_path_clears_pending_without_grant() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    requirement = _requirement()
    pending = _pending(requirement)
    task.runtime.governance.agent_governance_hitl_pending = pending
    AgentGovernanceHumanApprovalGrantCoordinator.clear_pending_on_reject_or_escalate(
        task,
        checkpoint_store=store,
    )
    assert task.runtime.governance.agent_governance_hitl_pending is None
    assert task.runtime.governance.agent_governance_human_approval_grant is None
    reloaded = Task.model_validate(
        store.get_latest(str(task.task_id), task.tenant_id).task_snapshot
    )
    assert reloaded.runtime.governance.agent_governance_hitl_pending is None


def test_incompatible_existing_grant_fails_closed() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    requirement = _requirement()
    pending = _pending(requirement)
    _attach_resolution(task, pending)
    approver = local_development_approver_evidence(tenant_id=task.tenant_id)
    other_grant = _grant(task).model_copy(
        update={"agent_governance_invocation_scope_id": "agr_other_scope"},
    )
    from intergrax.runtime.human.agent_governance_grant_lifecycle import (
        TaskAgentGovernanceGrantLifecycleAdapter,
    )

    adapter = TaskAgentGovernanceGrantLifecycleAdapter(task=task, checkpoint_store=store)
    adapter.persist_available_grant(
        task_id=pending.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=0,
        grant=other_grant,
    )
    with pytest.raises(AgentGovernanceHumanApprovalGrantError):
        AgentGovernanceHumanApprovalGrantCoordinator.persist_available_grant_from_human_approve(
            task,
            checkpoint_store=store,
            approver=approver,
        )


def test_runtime_state_carries_verified_agent_governance_transport() -> None:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

    assert hasattr(RuntimeState, "__dataclass_fields__")
    assert "verified_agent_governance_human_approval" in RuntimeState.__dataclass_fields__
