# © Artur Czarnecki. All rights reserved.

"""W3-C3 — DECISION_DURABLE recovery admission wiring."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.decision_checkpoint import decision_checkpoint_state
from intergrax.contracts.decision_finalization import (
    decision_finalization_key,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import (
    initial_decision_lifecycle_state,
)
from intergrax.contracts.decision_revision import DecisionRevisionPolicy
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.recovery_admission import (
    RecoveryAdmissionOverloadMode,
    RecoveryAdmissionPolicy,
    RecoveryKind,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    save_decision_checkpoint,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
)
from intergrax.runtime.execution.in_memory_decision_checkpoint_persistence import (
    InMemoryDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.in_memory_decision_finalization_persistence import (
    InMemoryDecisionFinalizationPersistence,
)
from intergrax.runtime.resilience import decision_durable_recovery_handoff as handoff
from intergrax.runtime.resilience.decision_durable_recovery_handoff import (
    recovery_admission_request_for_decision_durable,
    resume_decision_from_durable_state_with_recovery_admission,
)
from intergrax.runtime.resilience.local_recovery_admission import LocalRecoveryAdmission

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=_lineage(),
    )


def _reject_policy(capacity: int) -> RecoveryAdmissionPolicy:
    return RecoveryAdmissionPolicy(
        max_concurrent_recovery_starts=capacity,
        overload_mode=RecoveryAdmissionOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


def _wait_policy(capacity: int, timeout: float) -> RecoveryAdmissionPolicy:
    return RecoveryAdmissionPolicy(
        max_concurrent_recovery_starts=capacity,
        overload_mode=RecoveryAdmissionOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=timeout,
    )


@pytest.mark.asyncio
async def test_admission_skipped_when_port_none() -> None:
    identity = _identity()
    key = decision_finalization_key(identity)
    checkpoint = decision_checkpoint_state(
        lifecycle=initial_decision_lifecycle_state(identity),
        finalization=initial_decision_finalize_guard(key),
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    finalization_store = InMemoryDecisionFinalizationPersistence[
        IncidentDecisionPayload
    ]()
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)

    resumed = await resume_decision_from_durable_state_with_recovery_admission(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=key,
        execution_lineage=identity.execution,
        recovery_admission=None,
    )
    assert resumed is not None


@pytest.mark.asyncio
async def test_exception_during_recovery_releases_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    key = decision_finalization_key(identity)
    admission = LocalRecoveryAdmission(
        {RecoveryKind.DECISION_DURABLE: _reject_policy(1)},
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    finalization_store = InMemoryDecisionFinalizationPersistence[
        IncidentDecisionPayload
    ]()

    def _boom(
        *,
        checkpoint_persistence: object,
        finalization_persistence: object,
        key: object,
        runtime_revision_policy: DecisionRevisionPolicy | None = None,
    ) -> None:
        raise RuntimeError("recovery failed")

    monkeypatch.setattr(handoff, "resume_decision_from_durable_state", _boom)

    with pytest.raises(RuntimeError, match="recovery failed"):
        await resume_decision_from_durable_state_with_recovery_admission(
            checkpoint_persistence=checkpoint_store,
            finalization_persistence=finalization_store,
            key=key,
            execution_lineage=identity.execution,
            recovery_admission=admission,
        )

    replacement = await admission.acquire(
        recovery_admission_request_for_decision_durable(
            key=key,
            execution_lineage=identity.execution,
        ),
    )
    await replacement.release()


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_permit_releases_no_slot() -> None:
    identity = _identity()
    key = decision_finalization_key(identity)
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    finalization_store = InMemoryDecisionFinalizationPersistence[
        IncidentDecisionPayload
    ]()
    admission = LocalRecoveryAdmission(
        {RecoveryKind.DECISION_DURABLE: _wait_policy(1, 30.0)},
    )
    holder = await admission.acquire(
        recovery_admission_request_for_decision_durable(
            key=key,
            execution_lineage=identity.execution,
        ),
    )
    waiter = asyncio.create_task(
        resume_decision_from_durable_state_with_recovery_admission(
            checkpoint_persistence=checkpoint_store,
            finalization_persistence=finalization_store,
            key=key,
            execution_lineage=identity.execution,
            recovery_admission=admission,
        ),
    )
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await holder.release()
    resumed = await resume_decision_from_durable_state_with_recovery_admission(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=key,
        execution_lineage=identity.execution,
        recovery_admission=admission,
    )
    assert resumed is None
