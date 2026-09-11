# © Artur Czarnecki. All rights reserved.

"""Canonical DECISION_DURABLE recovery entry for qualification workers (W3-C4)."""

from __future__ import annotations

import asyncio
from typing import TypeVar

from intergrax.contracts.decision_checkpoint import DecisionCheckpointState
from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.contracts.decision_identity import DecisionExecutionLineage
from intergrax.contracts.decision_revision import DecisionRevisionPolicy
from intergrax.contracts.recovery_admission import (
    RecoveryAdmissionOverloadMode,
    RecoveryAdmissionPolicy,
    RecoveryKind,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionFinalizationPersistence,
)
from intergrax.runtime.resilience.decision_durable_recovery_handoff import (
    resume_decision_from_durable_state_with_recovery_admission,
)
from intergrax.runtime.resilience.local_recovery_admission import LocalRecoveryAdmission

T = TypeVar("T")

_DEFAULT_DECISION_DURABLE_START_CAPACITY = 32


def qualification_decision_durable_recovery_admission() -> LocalRecoveryAdmission:
    """Process-local recovery admission for DS-E2E qualification workers."""
    return LocalRecoveryAdmission(
        {
            RecoveryKind.DECISION_DURABLE: RecoveryAdmissionPolicy(
                max_concurrent_recovery_starts=_DEFAULT_DECISION_DURABLE_START_CAPACITY,
                overload_mode=RecoveryAdmissionOverloadMode.REJECT,
                wait_timeout_seconds=None,
            ),
        },
    )


def resume_decision_durable_with_canonical_recovery_admission(
    *,
    checkpoint_persistence: DecisionCheckpointPersistence[T],
    finalization_persistence: DecisionFinalizationPersistence[T],
    key: DecisionFinalizationKey,
    execution_lineage: DecisionExecutionLineage,
    runtime_revision_policy: DecisionRevisionPolicy | None = None,
) -> DecisionCheckpointState[T] | None:
    """Run durable decision recovery through the canonical admission handoff."""
    admission = qualification_decision_durable_recovery_admission()

    async def _run() -> DecisionCheckpointState[T] | None:
        return await resume_decision_from_durable_state_with_recovery_admission(
            checkpoint_persistence=checkpoint_persistence,
            finalization_persistence=finalization_persistence,
            key=key,
            execution_lineage=execution_lineage,
            recovery_admission=admission,
            runtime_revision_policy=runtime_revision_policy,
        )

    return asyncio.run(_run())
