# © Artur Czarnecki. All rights reserved.

"""W3-C4 — DECISION_DURABLE recovery admission qualification matrix."""

from __future__ import annotations

import asyncio
import time

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
from intergrax.contracts.decision_lifecycle import initial_decision_lifecycle_state
from intergrax.contracts.decision_revision import DecisionRevisionPolicy
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.recovery_admission import (
    RecoveryAdmissionExceededError,
    RecoveryAdmissionOverloadMode,
    RecoveryAdmissionPolicy,
    RecoveryKind,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    StaleDecisionCheckpointWriteError,
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
from intergrax.contracts.decision_event_append import StaleDecisionEventAppendError
from intergrax.runtime.resilience import decision_durable_recovery_handoff as handoff
from intergrax.runtime.resilience.decision_durable_recovery_handoff import (
    recovery_admission_request_for_decision_durable,
    resume_decision_from_durable_state_with_recovery_admission,
)
from intergrax.runtime.resilience.local_recovery_admission import LocalRecoveryAdmission

_RELEASE_POLL_INTERVAL_SECONDS = 0.05

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
        scope=DecisionScope(namespace="incident", subject="incident-w3c4"),
        tenant_id="tenant-w3c4",
        execution=_lineage(),
    )


def _reject_policy(capacity: int) -> RecoveryAdmissionPolicy:
    return RecoveryAdmissionPolicy(
        max_concurrent_recovery_starts=capacity,
        overload_mode=RecoveryAdmissionOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


def _stores() -> tuple[
    InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload],
    InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload],
]:
    return (
        InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload](),
        InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload](),
    )


@pytest.mark.asyncio
async def test_concurrent_recovery_storm_caps_inflight_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = 5
    storm_size = 100
    admission = LocalRecoveryAdmission(
        {RecoveryKind.DECISION_DURABLE: _reject_policy(capacity)},
    )
    identity = _identity()
    key = decision_finalization_key(identity)
    checkpoint_store, finalization_store = _stores()
    checkpoint = decision_checkpoint_state(
        lifecycle=initial_decision_lifecycle_state(identity),
        finalization=initial_decision_finalize_guard(key),
    )
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)
    start_barrier = asyncio.Barrier(storm_size)
    errors: list[RecoveryAdmissionExceededError] = []
    lock = asyncio.Lock()

    def _slow_impl(
        *,
        checkpoint_persistence: object,
        finalization_persistence: object,
        key: object,
        runtime_revision_policy: DecisionRevisionPolicy | None = None,
    ) -> object:
        time.sleep(0.05)
        return checkpoint

    monkeypatch.setattr(handoff, "_resume_decision_from_durable_state_impl", _slow_impl)

    async def contender() -> None:
        await start_barrier.wait()
        try:
            await resume_decision_from_durable_state_with_recovery_admission(
                checkpoint_persistence=checkpoint_store,
                finalization_persistence=finalization_store,
                key=key,
                execution_lineage=identity.execution,
                recovery_admission=admission,
            )
        except RecoveryAdmissionExceededError as exc:
            async with lock:
                errors.append(exc)

    await asyncio.wait_for(
        asyncio.gather(*[asyncio.create_task(contender()) for _ in range(storm_size)]),
        timeout=30.0,
    )
    assert len(errors) == storm_size - capacity


@pytest.mark.asyncio
async def test_cancel_during_running_recovery_releases_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    key = decision_finalization_key(identity)
    admission = LocalRecoveryAdmission(
        {RecoveryKind.DECISION_DURABLE: _reject_policy(1)},
    )
    checkpoint_store, finalization_store = _stores()

    def _slow_sync(
        *,
        checkpoint_persistence: object,
        finalization_persistence: object,
        key: object,
        runtime_revision_policy: DecisionRevisionPolicy | None = None,
    ) -> None:
        time.sleep(2.0)

    monkeypatch.setattr(handoff, "_resume_decision_from_durable_state_impl", _slow_sync)

    runner = asyncio.create_task(
        resume_decision_from_durable_state_with_recovery_admission(
            checkpoint_persistence=checkpoint_store,
            finalization_persistence=finalization_store,
            key=key,
            execution_lineage=identity.execution,
            recovery_admission=admission,
        ),
    )
    await asyncio.sleep(0.05)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    for _ in range(100):
        try:
            replacement = await admission.acquire(
                recovery_admission_request_for_decision_durable(
                    key=key,
                    execution_lineage=identity.execution,
                ),
            )
        except RecoveryAdmissionExceededError:
            await asyncio.sleep(_RELEASE_POLL_INTERVAL_SECONDS)
            continue
        await replacement.release()
        return
    raise AssertionError("permit not released after cancellation")


@pytest.mark.asyncio
async def test_snapshot_cas_conflict_releases_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    key = decision_finalization_key(identity)
    admission = LocalRecoveryAdmission(
        {RecoveryKind.DECISION_DURABLE: _reject_policy(1)},
    )
    checkpoint_store, finalization_store = _stores()

    def _stale_snapshot(
        *,
        checkpoint_persistence: object,
        finalization_persistence: object,
        key: object,
        runtime_revision_policy: DecisionRevisionPolicy | None = None,
    ) -> None:
        raise StaleDecisionCheckpointWriteError("snapshot revision 10 stale vs 11")

    monkeypatch.setattr(handoff, "_resume_decision_from_durable_state_impl", _stale_snapshot)

    with pytest.raises(StaleDecisionCheckpointWriteError):
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
async def test_event_append_conflict_releases_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    key = decision_finalization_key(identity)
    admission = LocalRecoveryAdmission(
        {RecoveryKind.DECISION_DURABLE: _reject_policy(1)},
    )
    checkpoint_store, finalization_store = _stores()

    def _stale_event_append(
        *,
        checkpoint_persistence: object,
        finalization_persistence: object,
        key: object,
        runtime_revision_policy: DecisionRevisionPolicy | None = None,
    ) -> None:
        raise StaleDecisionEventAppendError("expected sequence 20 got 21")

    monkeypatch.setattr(handoff, "_resume_decision_from_durable_state_impl", _stale_event_append)

    with pytest.raises(StaleDecisionEventAppendError):
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
