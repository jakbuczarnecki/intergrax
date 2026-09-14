# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral dual-read resolution for background execution identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from intergrax.contracts.npsc5f_compatibility import (
    AmbiguousLegacyExecutionIdentityError,
    BackgroundExecutionIdentityConflictError,
    LegacyBackgroundExecutionIdentityIncompatibleError,
)
from intergrax.runtime.background_execution.identity_types import (
    PersistedBackgroundExecutionIdentity,
)
from intergrax.runtime.background_execution.identity_record_codec import (
    DecodedBackgroundIdentityRecord,
    same_identity_triplet,
)
from intergrax.runtime.observability.causal_evidence_enrichment import (
    CanonicalExecutionIdLookupPort,
)


def _to_persisted(
    decoded: DecodedBackgroundIdentityRecord,
) -> PersistedBackgroundExecutionIdentity:
    if decoded.execution_id is None:
        raise LegacyBackgroundExecutionIdentityIncompatibleError(
            "legacy background identity missing ExecutionId",
        )
    return PersistedBackgroundExecutionIdentity(
        task_id=decoded.task_id,
        run_id=decoded.run_id,
        attempt_id=decoded.attempt_id,
        execution_id=decoded.execution_id,
    )


def reconcile_dual_read_records(
    *,
    v2_candidate: DecodedBackgroundIdentityRecord | None,
    v1_candidate: DecodedBackgroundIdentityRecord | None,
    tenant_id: str,
    lookup: CanonicalExecutionIdLookupPort | None = None,
) -> PersistedBackgroundExecutionIdentity:
    """Read v2 first, legacy v1 fallback, fail closed on conflict or ambiguity."""
    if v2_candidate is not None and v2_candidate.kind == "complete_v2":
        complete_v2 = _to_persisted(v2_candidate)
        if v1_candidate is not None and v1_candidate.kind == "legacy_v1":
            if not same_identity_triplet(v1_candidate, v2_candidate):
                raise BackgroundExecutionIdentityConflictError(
                    "v1 and v2 background identity triplets disagree",
                )
            if v2_candidate.execution_id is None:
                raise BackgroundExecutionIdentityConflictError(
                    "v2 background identity record missing ExecutionId",
                )
        return complete_v2

    if v1_candidate is None:
        if v2_candidate is None:
            raise LegacyBackgroundExecutionIdentityIncompatibleError(
                "background execution identity not found",
            )
        raise LegacyBackgroundExecutionIdentityIncompatibleError(
            "background execution identity record is not v2-complete",
        )

    legacy = v1_candidate
    if lookup is None:
        raise LegacyBackgroundExecutionIdentityIncompatibleError(
            "legacy v1 background identity cannot resolve ExecutionId without lookup port",
        )
    lookup_result = lookup.lookup_execution_id(
        tenant_id=tenant_id,
        task_id=legacy.task_id,
        run_id=legacy.run_id,
        attempt_id=legacy.attempt_id,
    )
    if lookup_result.outcome == "unresolved":
        raise LegacyBackgroundExecutionIdentityIncompatibleError(
            "legacy v1 background identity ExecutionId lookup unresolved",
        )
    if lookup_result.outcome == "ambiguous":
        raise AmbiguousLegacyExecutionIdentityError(
            "legacy v1 background identity ExecutionId lookup ambiguous",
        )
    if lookup_result.execution_id is None:
        raise LegacyBackgroundExecutionIdentityIncompatibleError(
            "canonical lookup returned resolved without ExecutionId",
        )
    return PersistedBackgroundExecutionIdentity(
        task_id=legacy.task_id,
        run_id=legacy.run_id,
        attempt_id=legacy.attempt_id,
        execution_id=lookup_result.execution_id,
    )


class BackgroundExecutionIdentityMigrationPort(Protocol):
    """Provider-owned migrate-on-read hook (optional, idempotent)."""

    def persist_v2_if_absent(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        identity: PersistedBackgroundExecutionIdentity,
    ) -> None:
        """Write v2 durable identity without overwriting conflicting v2."""


@dataclass(frozen=True, slots=True)
class BackgroundIdentityMigrationAudit:
    source_version: Literal["legacy_v1", "complete_v2"]
    target_version: Literal["intergrax.bg_exec_identity.v2"]
    migration_method: Literal["migrate_on_read", "native_v2"]
    outcome: Literal["resolved", "conflict", "incompatible"]
