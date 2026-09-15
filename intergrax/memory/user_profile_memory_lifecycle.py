# © Artur Czarnecki. All rights reserved.

"""User profile memory lifecycle coordinator (primary vs derived projections)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryLifecycleOperation,
    MemoryLifecycleOutcome,
    MemoryProjectionOperation,
    MemoryProjectionOperationEvidence,
    MemoryReconciliationDisposition,
    MemoryReconciliationOutcome,
    UserProfileMemoryProjection,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.memory_projection_failure import classify_memory_projection_failure
from intergrax.memory.memory_temporal import filter_active_memory_entries
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
    ) -> None:
        self._projections = tuple(projections)

    @property
    def projections(self) -> tuple[UserProfileMemoryProjection, ...]:
        return self._projections

    async def apply_after_primary_upsert(
        self,
        *,
        operation: MemoryLifecycleOperation,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> MemoryLifecycleOutcome:
        evidence = await self._run_projection_upsert(user_id=user_id, entry=entry)
        disposition = self._disposition_from_evidence(evidence)
        return MemoryLifecycleOutcome(
            operation=operation,
            disposition=disposition,
            user_id=user_id,
            memory_entity_ids=(entry.entry_id,),
            primary_applied=True,
            projection_evidence=evidence,
        )

    async def apply_after_primary_deletes(
        self,
        *,
        operation: MemoryLifecycleOperation,
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
        evidence = await self._run_projection_delete(entry_ids=ids)
        disposition = self._disposition_from_evidence(evidence)
        return MemoryLifecycleOutcome(
            operation=operation,
            disposition=disposition,
            user_id=user_id,
            memory_entity_ids=ids,
            primary_applied=True,
            projection_evidence=evidence,
        )

    async def reconcile_user(
        self,
        *,
        user_id: str,
        profile: UserProfile | None,
    ) -> MemoryReconciliationOutcome:
        if profile is None:
            active_ids: frozenset[str] = frozenset()
        else:
            active_ids = active_memory_entry_ids(profile)
        context = UserProfileMemoryReconciliationContext(
            user_id=user_id,
            profile=profile,
            authoritative_active_entry_ids=active_ids,
        )
        evidence: list[MemoryProjectionOperationEvidence] = []
        repaired = False
        for projection in self._projections:
            try:
                await projection.reconcile(context)
                evidence.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.REBUILD,
                        succeeded=True,
                    )
                )
                repaired = True
            except BaseException as exc:
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
        elif repaired and self._projections:
            disposition = MemoryReconciliationDisposition.REPAIRED
        else:
            disposition = MemoryReconciliationDisposition.CONSISTENT
        return MemoryReconciliationOutcome(
            user_id=user_id,
            disposition=disposition,
            projection_evidence=tuple(evidence),
        )

    async def _run_projection_upsert(
        self,
        *,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> tuple[MemoryProjectionOperationEvidence, ...]:
        if not self._projections:
            return ()
        results: list[MemoryProjectionOperationEvidence] = []
        for projection in self._projections:
            try:
                await projection.upsert_memory_entry(user_id, entry)
                results.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.UPSERT,
                        succeeded=True,
                    )
                )
            except BaseException as exc:
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
        entry_ids: Sequence[str],
    ) -> tuple[MemoryProjectionOperationEvidence, ...]:
        if not self._projections:
            return ()
        results: list[MemoryProjectionOperationEvidence] = []
        for projection in self._projections:
            try:
                await projection.delete_memory_entries(entry_ids)
                results.append(
                    MemoryProjectionOperationEvidence(
                        projection_id=projection.projection_id,
                        operation=MemoryProjectionOperation.DELETE,
                        succeeded=True,
                    )
                )
            except BaseException as exc:
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
