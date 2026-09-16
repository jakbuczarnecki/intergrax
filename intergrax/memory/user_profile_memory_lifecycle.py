# © Artur Czarnecki. All rights reserved.

"""User profile memory lifecycle coordinator (primary vs derived projections)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryLifecycleOperation,
    MemoryLifecycleOutcome,
    MemoryProjectionOperation,
    MemoryProjectionOperationEvidence,
    MemoryProjectionReconciliationDisposition,
    MemoryReconciliationDisposition,
    MemoryReconciliationOutcome,
    UserProfileMemoryProjection,
    UserProfileMemoryReconciliationContext,
    user_profile_memory_projection_context,
)
from intergrax.memory.memory_projection_failure import classify_memory_projection_failure
from intergrax.memory.memory_temporal import filter_active_memory_entries
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    default_memory_diagnostic_emitter,
)
from intergrax.memory.memory_observability_support import (
    emit_lifecycle_terminal,
    emit_reconciliation_terminal,
)
from intergrax.memory.user_profile_memory import UserProfile, UserProfileMemoryEntry

__all__ = [
    "UserProfileMemoryLifecycleCoordinator",
    "UserProfileMemoryLifecyclePartialError",
    "active_memory_entry_ids",
]


class UserProfileMemoryLifecyclePartialError(RuntimeError):
    """Primary mutation succeeded but one or more projections failed."""

    def __init__(self, outcome: MemoryLifecycleOutcome) -> None:
        super().__init__(
            f"memory lifecycle partial failure: operation={outcome.operation.value}, "
            f"user_id={outcome.user_id}"
        )
        self.outcome = outcome


def active_memory_entry_ids(profile: UserProfile) -> frozenset[str]:
    return frozenset(entry.entry_id for entry in filter_active_memory_entries(profile.memory_entries))


class UserProfileMemoryLifecycleCoordinator:
    def __init__(
        self,
        *,
        projections: Sequence[UserProfileMemoryProjection],
        diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._projections = tuple(projections)
        self._diagnostic_emitter = diagnostic_emitter or default_memory_diagnostic_emitter()
        self._tenant_id = tenant_id

    @property
    def projections(self) -> tuple[UserProfileMemoryProjection, ...]:
        return self._projections

    async def apply_after_primary_upsert(
        self,
        *,
        operation: MemoryLifecycleOperation,
        identity: RequestIdentity,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> MemoryLifecycleOutcome:
        evidence = await self._run_projection_upsert(identity=identity, entry=entry)
        disposition = self._disposition_from_evidence(evidence)
        result = MemoryLifecycleOutcome(
            operation=operation,
            disposition=disposition,
            user_id=user_id,
            memory_entity_ids=(entry.entry_id,),
            primary_applied=True,
            projection_evidence=evidence,
        )
        emit_lifecycle_terminal(
            self._diagnostic_emitter,
            user_id=user_id,
            tenant_id=self._tenant_id,
            outcome=result,
        )
        return result

    async def apply_after_primary_deletes(
        self,
        *,
        operation: MemoryLifecycleOperation,
        identity: RequestIdentity,
        user_id: str,
        entry_ids: Sequence[str],
    ) -> MemoryLifecycleOutcome:
        ids = tuple(entry_id for entry_id in entry_ids if entry_id)
        if not ids:
            return MemoryLifecycleOutcome(
                operation=operation,
                disposition=MemoryLifecycleDisposition.UNCHANGED,
                user_id=user_id,
                memory_entity_ids=(),
                primary_applied=True,
                projection_evidence=(),
            )
        evidence = await self._run_projection_delete(identity=identity, entry_ids=ids)
        disposition = self._disposition_from_evidence(evidence)
        result = MemoryLifecycleOutcome(
            operation=operation,
            disposition=disposition,
            user_id=user_id,
            memory_entity_ids=ids,
            primary_applied=True,
            projection_evidence=evidence,
        )
        emit_lifecycle_terminal(
            self._diagnostic_emitter,
            user_id=user_id,
            tenant_id=self._tenant_id,
            outcome=result,
        )
        return result

    async def reconcile_user(
        self,
        *,
        identity: RequestIdentity,
        profile: UserProfile | None,
    ) -> MemoryReconciliationOutcome:
        user_id = identity.user_id or ""
        if profile is None:
            active_ids: frozenset[str] = frozenset()
        else:
            active_ids = active_memory_entry_ids(profile)
        context = UserProfileMemoryReconciliationContext(
            identity=identity,
            profile=profile,
            authoritative_active_entry_ids=active_ids,
        )
        evidence: list[MemoryProjectionOperationEvidence] = []
        any_repaired = False
        for projection in self._projections:
            try:
                result = await projection.reconcile(context)
                evidence.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.REBUILD,
                        succeeded=True,
                    )
                )
                if (
                    result.disposition
                    is MemoryProjectionReconciliationDisposition.REPAIRED
                ):
                    any_repaired = True
            except Exception as exc:
                failure = classify_memory_projection_failure(
                    projection_id=projection.projection_id,
                    operation=MemoryProjectionOperation.REBUILD,
                    exc=exc,
                )
                evidence.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.REBUILD,
                        succeeded=False,
                        failure=failure,
                    )
                )
        if any(not item.succeeded for item in evidence):
            disposition = MemoryReconciliationDisposition.FAILED
        elif any_repaired:
            disposition = MemoryReconciliationDisposition.REPAIRED
        else:
            disposition = MemoryReconciliationDisposition.CONSISTENT
        result = MemoryReconciliationOutcome(
            user_id=user_id,
            disposition=disposition,
            projection_evidence=tuple(evidence),
        )
        emit_reconciliation_terminal(
            self._diagnostic_emitter,
            tenant_id=self._tenant_id,
            user_id=user_id,
            outcome=result,
        )
        return result

    async def _run_projection_upsert(
        self,
        *,
        identity: RequestIdentity,
        entry: UserProfileMemoryEntry,
    ) -> tuple[MemoryProjectionOperationEvidence, ...]:
        if not self._projections:
            return ()
        context = user_profile_memory_projection_context(identity)
        results: list[MemoryProjectionOperationEvidence] = []
        for projection in self._projections:
            try:
                await projection.upsert_memory_entry(context, entry)
                results.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.UPSERT,
                        succeeded=True,
                    )
                )
            except Exception as exc:
                failure = classify_memory_projection_failure(
                    projection_id=projection.projection_id,
                    operation=MemoryProjectionOperation.UPSERT,
                    exc=exc,
                )
                results.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.UPSERT,
                        succeeded=False,
                        failure=failure,
                    )
                )
        return tuple(results)

    async def _run_projection_delete(
        self,
        *,
        identity: RequestIdentity,
        entry_ids: Sequence[str],
    ) -> tuple[MemoryProjectionOperationEvidence, ...]:
        if not self._projections:
            return ()
        context = user_profile_memory_projection_context(identity)
        results: list[MemoryProjectionOperationEvidence] = []
        for projection in self._projections:
            try:
                await projection.delete_memory_entries(context, entry_ids)
                results.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.DELETE,
                        succeeded=True,
                    )
                )
            except Exception as exc:
                failure = classify_memory_projection_failure(
                    projection_id=projection.projection_id,
                    operation=MemoryProjectionOperation.DELETE,
                    exc=exc,
                )
                results.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.DELETE,
                        succeeded=False,
                        failure=failure,
                    )
                )
        return tuple(results)

    @staticmethod
    def _disposition_from_evidence(
        evidence: tuple[MemoryProjectionOperationEvidence, ...],
    ) -> MemoryLifecycleDisposition:
        if not evidence:
            return MemoryLifecycleDisposition.CONSISTENT
        if any(not item.succeeded for item in evidence):
            return MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
        return MemoryLifecycleDisposition.CONSISTENT

    def raise_if_partial(self, outcome: MemoryLifecycleOutcome) -> None:
        if outcome.requires_reconciliation:
            raise UserProfileMemoryLifecyclePartialError(outcome)
