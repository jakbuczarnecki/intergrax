# © Artur Czarnecki. All rights reserved.

"""Typed memory retention sweep (PDC-D2-R015, ADR-PDC-02).

The sweep owns no scheduler and no storage. It decides *which* USER memory entries have outlived
their retention window and enforces expiry through the single memory owner, ``MemoryControlPlane``
(``forget`` with the retention path), which soft-deletes the primary record and tombstones the
matching vector-index entry through the lifecycle projections. Repeating the sweep is idempotent.

Background execution reuses the platform task runtime: ``register_memory_retention_sweep_task``
registers a ``TaskDefinition`` that the existing ``WorkerRuntime`` admits, identifies and records.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.memory.contracts.memory_control import (
    MemoryControlBackendError,
    MemoryControlForgetRequest,
    MemoryControlNotFound,
    MemoryControlPartialLifecycleError,
    MemoryControlPlane,
    MemoryControlReconcileRequest,
    MemoryRetentionCandidateReader,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionOperation,
    MemoryReconciliationDisposition,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
    MemoryRetentionAction,
    MemoryRetentionPolicy,
    MemorySecurityContext,
)
from intergrax.memory.memory_temporal import _parse_iso_timestamp, is_memory_entry_expired
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.queueing.contracts.task_queue import TaskRequest
from intergrax.utils.time_provider import SystemTimeProvider

__all__ = [
    "MEMORY_RETENTION_SWEEP_TASK_NAME",
    "MemoryRetentionSweep",
    "MemoryRetentionSweepOutput",
    "MemoryRetentionSweepPayload",
    "MemoryRetentionSweepReport",
    "MemoryRetentionSweepRequest",
    "memory_retention_sweep_task_request",
    "register_memory_retention_sweep_task",
]

MEMORY_RETENTION_SWEEP_TASK_NAME = "memory.retention_sweep.v1"
DEFAULT_SWEEP_BATCH_LIMIT = 500


@dataclass(frozen=True, slots=True)
class MemoryRetentionSweepRequest:
    """Trusted identity (composition-owned) plus the evaluation instant."""

    identity: RequestIdentity
    as_of: datetime | None = None
    batch_limit: int = DEFAULT_SWEEP_BATCH_LIMIT

    def __post_init__(self) -> None:
        if self.batch_limit < 1:
            raise ValueError("batch_limit must be >= 1")
        if self.as_of is not None and self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")


@dataclass(frozen=True, slots=True)
class MemoryRetentionSweepReport:
    """Outcome of one sweep; ids only, never memory content."""

    tenant_id: str
    user_id: str
    as_of_iso: str
    scanned: int
    due: int
    forgotten_entry_ids: tuple[str, ...] = ()
    tombstoned_vector_entry_ids: tuple[str, ...] = ()
    partial_entry_ids: tuple[str, ...] = ()
    failed_entry_ids: tuple[str, ...] = ()
    reconciled: bool = False
    deferred: int = field(default=0)
    """Due entries left for the next run because ``batch_limit`` was reached."""


def _expiry_instant(
    entry: UserProfileMemoryEntry,
    *,
    as_of: datetime,
    identity: RequestIdentity,
    policy: MemoryRetentionPolicy | None,
) -> datetime | None:
    """Return the expiry instant when the entry is due at ``as_of``, else None."""
    if is_memory_entry_expired(entry, as_of=as_of):
        return _parse_iso_timestamp(entry.valid_until or "")
    if policy is None:
        return None
    scope = user_memory_scope(identity)
    decision = policy.evaluate(
        MemoryGovernanceEvaluationRequest(
            context=MemorySecurityContext(
                identity=identity,
                scope=scope,
                operation=MemoryGovernanceOperation.DELETE,
                reference_time=as_of,
            ),
            target=MemoryGovernanceTarget(memory_id=entry.entry_id),
            existing_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(entry),
        )
    )
    if decision.action is not MemoryRetentionAction.EXPIRE_AT or not decision.expire_at_iso:
        return None
    expire_at = _parse_iso_timestamp(decision.expire_at_iso)
    if expire_at is not None and as_of >= expire_at:
        return expire_at
    return None


class MemoryRetentionSweep:
    """Enforces retention expiry for one USER scope through ``MemoryControlPlane``."""

    def __init__(
        self,
        *,
        control_plane: MemoryControlPlane,
        candidates: MemoryRetentionCandidateReader,
        retention_policy: MemoryRetentionPolicy | None = None,
    ) -> None:
        self._plane = control_plane
        self._candidates = candidates
        self._policy = retention_policy

    async def run(self, request: MemoryRetentionSweepRequest) -> MemoryRetentionSweepReport:
        identity = request.identity
        scope = user_memory_scope(identity)
        user_id = scope.user_id or ""
        as_of = request.as_of or SystemTimeProvider.utc_now()
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)

        entries = await self._candidates.list_retention_candidates(user_id)
        due: list[tuple[datetime, str]] = []
        for entry in entries:
            instant = _expiry_instant(entry, as_of=as_of, identity=identity, policy=self._policy)
            if instant is not None:
                due.append((instant, entry.entry_id))
        due.sort()
        batch = due[: request.batch_limit]

        forgotten: list[str] = []
        tombstoned: list[str] = []
        partial: list[str] = []
        failed: list[str] = []
        for _instant, entry_id in batch:
            try:
                result = await self._plane.forget(
                    identity,
                    scope,
                    MemoryControlForgetRequest(entry_id=entry_id, include_expired=True),
                )
            except MemoryControlNotFound:
                continue  # already forgotten by a concurrent sweep: idempotent
            except MemoryControlPartialLifecycleError:
                partial.append(entry_id)
                continue
            except (MemoryControlBackendError, PermissionError):
                failed.append(entry_id)
                continue
            forgotten.append(entry_id)
            lifecycle = result.lifecycle
            if lifecycle is not None and any(
                evidence.operation is MemoryProjectionOperation.DELETE and evidence.succeeded
                for evidence in lifecycle.projection_evidence
            ):
                tombstoned.append(entry_id)

        reconciled = False
        if partial:
            # Primary delete applied but a projection failed: repair through the same owner.
            try:
                reconcile = await self._plane.reconcile(identity, scope, MemoryControlReconcileRequest())
            except (MemoryControlBackendError, MemoryControlPartialLifecycleError):
                reconcile = None
            outcome = reconcile.reconciliation if reconcile is not None else None
            if outcome is not None and outcome.disposition is not MemoryReconciliationDisposition.FAILED:
                reconciled = True
                tombstoned.extend(partial)

        return MemoryRetentionSweepReport(
            tenant_id=identity.tenant_id,
            user_id=user_id,
            as_of_iso=as_of.isoformat(),
            scanned=len(entries),
            due=len(due),
            forgotten_entry_ids=tuple(forgotten),
            tombstoned_vector_entry_ids=tuple(tombstoned),
            partial_entry_ids=tuple(partial),
            failed_entry_ids=tuple(failed),
            reconciled=reconciled,
            deferred=max(0, len(due) - len(batch)),
        )


# ---------------------------------------------------------------- background admission


class MemoryRetentionSweepPayload(BaseModel):
    """Wire payload of the background task; carries no authority (identity is resolved)."""

    tenant_id: str
    user_id: str
    as_of_iso: str | None = None
    batch_limit: int = Field(default=DEFAULT_SWEEP_BATCH_LIMIT, ge=1)


class MemoryRetentionSweepOutput(BaseModel):
    tenant_id: str
    user_id: str
    scanned: int
    due: int
    forgotten: int
    tombstoned_vectors: int
    partial: int
    failed: int
    reconciled: bool
    deferred: int


def memory_retention_sweep_task_request(
    *,
    tenant_id: str,
    run_id: str,
    user_id: str,
    as_of: datetime | None = None,
    batch_limit: int = DEFAULT_SWEEP_BATCH_LIMIT,
    idempotency_key: str | None = None,
) -> TaskRequest:
    """Build the ``TaskRequest`` that admits a sweep through the existing task queue."""
    payload = MemoryRetentionSweepPayload(
        tenant_id=tenant_id,
        user_id=user_id,
        as_of_iso=as_of.isoformat() if as_of is not None else None,
        batch_limit=batch_limit,
    )
    return TaskRequest(
        tenant_id=tenant_id,
        run_id=run_id,
        task_name=MEMORY_RETENTION_SWEEP_TASK_NAME,
        payload=payload.model_dump_json().encode("utf-8"),
        idempotency_key=idempotency_key,
    )


def _run_coroutine_blocking(coroutine: Any) -> Any:
    """Run ``coroutine`` to completion from a synchronous worker handler."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = asyncio.run(coroutine)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller thread
            box["error"] = exc

    thread = threading.Thread(target=_target, name="memory-retention-sweep")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def register_memory_retention_sweep_task(
    registry: TaskRegistry,
    *,
    sweep: MemoryRetentionSweep,
    identity_resolver: Callable[[str, str], RequestIdentity],
    clock: Callable[[], datetime] | None = None,
) -> TaskDefinition:
    """Register the sweep with the platform task registry (no scheduler is introduced).

    ``identity_resolver(tenant_id, user_id)`` is composition-owned trusted authority; the payload
    tenant must equal the execution identity tenant admitted by the worker runtime.
    """

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None = None,
        execution_identity: Any = None,
    ) -> ToolExecutionResult[BaseModel]:
        _ = run_id, idempotency_key
        try:
            body = MemoryRetentionSweepPayload.model_validate(json.loads(payload.decode("utf-8")))
        except Exception:
            return ToolExecutionResult.fail(
                "memory_retention_sweep.invalid_payload",
                "retention sweep payload is not valid",
            )
        admitted_tenant = getattr(execution_identity, "tenant_id", tenant_id)
        if body.tenant_id != tenant_id or body.tenant_id != admitted_tenant:
            return ToolExecutionResult.fail(
                "memory_retention_sweep.tenant_mismatch",
                "retention sweep payload tenant does not match the admitted execution tenant",
            )
        identity = identity_resolver(body.tenant_id, body.user_id)
        if identity.tenant_id != body.tenant_id or (identity.user_id or "") != body.user_id:
            return ToolExecutionResult.fail(
                "memory_retention_sweep.identity_mismatch",
                "resolved identity does not match the retention sweep payload",
            )
        as_of = _parse_iso_timestamp(body.as_of_iso) if body.as_of_iso else (clock() if clock is not None else None)
        report = _run_coroutine_blocking(
            sweep.run(
                MemoryRetentionSweepRequest(
                    identity=identity,
                    as_of=as_of,
                    batch_limit=body.batch_limit,
                )
            )
        )
        return ToolExecutionResult.ok(
            MemoryRetentionSweepOutput(
                tenant_id=report.tenant_id,
                user_id=report.user_id,
                scanned=report.scanned,
                due=report.due,
                forgotten=len(report.forgotten_entry_ids),
                tombstoned_vectors=len(report.tombstoned_vector_entry_ids),
                partial=len(report.partial_entry_ids),
                failed=len(report.failed_entry_ids),
                reconciled=report.reconciled,
                deferred=report.deferred,
            )
        )

    definition = TaskDefinition(
        task_name=MEMORY_RETENTION_SWEEP_TASK_NAME,
        payload_schema=MemoryRetentionSweepPayload,
        handler=handler,
    )
    registry.register(definition)
    return definition
