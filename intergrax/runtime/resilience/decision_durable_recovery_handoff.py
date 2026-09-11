# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DECISION_DURABLE recovery start handoff (W3-C §3.3).

Acquires a recovery-start permit before durable decision state materialization and
releases it after load completes (start-only; not execution lifecycle).
"""

from __future__ import annotations

import asyncio
from typing import TypeVar

from intergrax.contracts.decision_checkpoint import DecisionCheckpointState
from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.contracts.decision_identity import DecisionExecutionLineage
from intergrax.contracts.decision_revision import DecisionRevisionPolicy
from intergrax.contracts.recovery_admission import (
    RecoveryAdmissionPort,
    RecoveryAdmissionRequest,
    RecoveryKind,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionFinalizationPersistence,
)
from intergrax.runtime.execution.decision_recovery import (
    _resume_decision_from_durable_state_impl,
)

T = TypeVar("T")


def recovery_admission_request_for_decision_durable(
    *,
    key: DecisionFinalizationKey,
    execution_lineage: DecisionExecutionLineage,
) -> RecoveryAdmissionRequest:
    """Build one DECISION_DURABLE admission request from finalize key + execution lineage."""
    if type(key) is not DecisionFinalizationKey:
        raise TypeError("key must be DecisionFinalizationKey")
    if type(execution_lineage) is not DecisionExecutionLineage:
        raise TypeError("execution_lineage must be DecisionExecutionLineage")
    return RecoveryAdmissionRequest(
        tenant_id=key.tenant_id,
        task_id=execution_lineage.task_id,
        run_id=execution_lineage.run_id,
        attempt_id=execution_lineage.attempt_id,
        recovery_kind=RecoveryKind.DECISION_DURABLE,
    )


async def resume_decision_from_durable_state_with_recovery_admission(
    *,
    checkpoint_persistence: DecisionCheckpointPersistence[T],
    finalization_persistence: DecisionFinalizationPersistence[T],
    key: DecisionFinalizationKey,
    execution_lineage: DecisionExecutionLineage,
    recovery_admission: RecoveryAdmissionPort | None = None,
    runtime_revision_policy: DecisionRevisionPolicy | None = None,
) -> DecisionCheckpointState[T] | None:
    """Materialize decision durable state under optional recovery start admission."""
    recovery_permit = None
    if recovery_admission is not None:
        recovery_permit = await recovery_admission.acquire(
            recovery_admission_request_for_decision_durable(
                key=key,
                execution_lineage=execution_lineage,
            ),
        )
    try:
        return await asyncio.to_thread(
            _resume_decision_from_durable_state_impl,
            checkpoint_persistence=checkpoint_persistence,
            finalization_persistence=finalization_persistence,
            key=key,
            runtime_revision_policy=runtime_revision_policy,
        )
    finally:
        if recovery_permit is not None:
            await recovery_permit.release()
