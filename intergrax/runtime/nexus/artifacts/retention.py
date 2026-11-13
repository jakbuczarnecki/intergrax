# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Retention class, TTL and the expiry sweep (PDC-D8-R008, R009, PDC-D6-R022).

The sweep is a handler on the existing canonical background registry
(``TaskExecutionRegistry``), run by ``execute_logical_task`` like any other background work. This
module adds no scheduler and no registry. Audit evidence and artifacts under legal hold are never
deleted by TTL: the sweeper skips them and the store refuses them as well.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel

from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.artifact_ref import ArtifactRetentionClass
from intergrax.contracts.task_artifacts import ArtifactRetentionPolicy
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.queueing.worker.registry import BackgroundTaskHandler, TaskExecutionRegistry
from intergrax.runtime.agent_governance.audit import GovernanceAuditRecorder
from intergrax.runtime.nexus.artifacts.governed_access import (
    CAPABILITY_DELETE,
    ArtifactAccessContext,
    ArtifactOperationAuditor,
)
from intergrax.runtime.nexus.artifacts.store_base import (
    ArtifactDeletionRefused,
    ArtifactStore,
    protection_reason,
)
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity

ARTIFACT_RETENTION_SWEEP_TASK_NAME = "intergrax.artifacts.retention_sweep"
SWEEP_ACTOR_ID = "artifact_retention_sweep"
DEFAULT_SWEEP_LIMIT = 1000


@dataclass(frozen=True)
class ArtifactRetentionSchedule:
    """TTL per retention class. ``None`` means the class never expires by TTL."""

    ttl_seconds_by_class: Mapping[ArtifactRetentionClass, int | None] = field(default_factory=dict)

    @classmethod
    def default(cls) -> ArtifactRetentionSchedule:
        return cls(
            {
                ArtifactRetentionClass.EPHEMERAL: 3600,
                ArtifactRetentionClass.DEFAULT: None,
                ArtifactRetentionClass.EXTENDED: None,
                ArtifactRetentionClass.AUDIT_EVIDENCE: None,
            }
        )

    def ttl_for(self, retention_class: ArtifactRetentionClass) -> int | None:
        return self.ttl_seconds_by_class.get(retention_class)


def resolve_expiry(
    created_at: datetime,
    *,
    retention_class: ArtifactRetentionClass,
    schedule: ArtifactRetentionSchedule | None = None,
    policy: ArtifactRetentionPolicy | None = None,
) -> datetime | None:
    """Expiry instant for a new artifact, or ``None`` when it does not expire.

    Audit evidence never expires. A ``retain_hours`` set on the existing
    ``ArtifactRetentionPolicy`` wins over the class schedule (that field now has a consumer).
    """
    if retention_class is ArtifactRetentionClass.AUDIT_EVIDENCE:
        return None
    if policy is not None and policy.retain_hours is not None:
        return created_at + timedelta(hours=policy.retain_hours)
    ttl = (schedule or ArtifactRetentionSchedule.default()).ttl_for(retention_class)
    return None if ttl is None else created_at + timedelta(seconds=ttl)


@dataclass(frozen=True)
class RetentionSweepReport:
    tenant_id: str
    examined: int
    deleted_ids: tuple[str, ...]
    protected_ids: tuple[str, ...]


class ArtifactRetentionSweeper:
    """Deletes expired, unprotected artifacts of one tenant; refuses evidence and legal hold."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        clock: Callable[[], datetime] | None = None,
        audit: GovernanceAuditRecorder | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or SystemTimeProvider.utc_now
        self._auditor = ArtifactOperationAuditor(audit) if audit is not None else None

    def sweep(
        self,
        tenant_id: str,
        *,
        context: ArtifactAccessContext | None = None,
        limit: int = DEFAULT_SWEEP_LIMIT,
    ) -> RetentionSweepReport:
        if context is not None and context.tenant_id != tenant_id:
            raise ValueError("sweep context tenant differs from the swept tenant")
        if self._auditor is not None and context is None:
            raise ValueError("a sweep with an audit recorder needs an access context")
        now = self._clock()
        candidates = self._store.list_expired(tenant_id, now, limit=limit)
        deleted: list[str] = []
        protected: list[str] = []
        for artifact in candidates:
            history_classes = [a.retention_class for a in self._store.list_versions(tenant_id, artifact.artifact_id)]
            reason = protection_reason(
                history_classes,
                legal_hold=self._store.is_under_legal_hold(tenant_id, artifact.artifact_id),
            )
            if reason is None:
                try:
                    self._store.delete_artifact(tenant_id, artifact.artifact_id)
                except ArtifactDeletionRefused as exc:  # raced with a hold or evidence marking
                    reason = exc.reason
                except KeyError:
                    continue  # already gone
            if reason is None:
                deleted.append(artifact.artifact_id)
                self._record(
                    context,
                    artifact.artifact_id,
                    ToolAuthorizationDecisionState.ALLOW,
                    "completed",
                    f"ttl_expired;expired_at={artifact.expires_at_utc}",
                    now,
                )
            else:
                protected.append(artifact.artifact_id)
                self._record(
                    context,
                    artifact.artifact_id,
                    ToolAuthorizationDecisionState.DENY,
                    "deletion_refused",
                    f"{reason};ttl_deletion_skipped",
                    now,
                )
        return RetentionSweepReport(
            tenant_id=tenant_id,
            examined=len(candidates),
            deleted_ids=tuple(deleted),
            protected_ids=tuple(protected),
        )

    def _record(self, context, artifact_id, decision, check, detail, at) -> None:
        if self._auditor is None or context is None:
            return
        self._auditor.record(
            context,
            capability=CAPABILITY_DELETE,
            artifact_id=artifact_id,
            decision=decision,
            check=check,
            detail=detail,
            at=at,
        )


class ArtifactRetentionSweepOutput(BaseModel):
    tenant_id: str
    examined: int
    deleted: int
    protected_retained: int


def make_retention_sweep_handler(sweeper: ArtifactRetentionSweeper) -> BackgroundTaskHandler:
    """Wrap the sweeper as a canonical ``BackgroundTaskHandler`` (tenant comes from the identity)."""

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[BaseModel]:
        limit = DEFAULT_SWEEP_LIMIT
        if payload:
            try:
                limit = int(json.loads(payload.decode("utf-8")).get("limit", limit))
            except (ValueError, AttributeError, UnicodeDecodeError):
                return ToolExecutionResult.fail("invalid_payload", "sweep payload must be JSON with an int limit")
        context = ArtifactAccessContext.for_background_execution(agent_id=SWEEP_ACTOR_ID, identity=execution_identity)
        report = sweeper.sweep(execution_identity.tenant_id, context=context, limit=limit)
        return ToolExecutionResult.ok(
            ArtifactRetentionSweepOutput(
                tenant_id=report.tenant_id,
                examined=report.examined,
                deleted=len(report.deleted_ids),
                protected_retained=len(report.protected_ids),
            )
        )

    return handler


def register_retention_sweep(registry: TaskExecutionRegistry, sweeper: ArtifactRetentionSweeper) -> None:
    """Admit the sweep as background work on the existing registry (one canonical handler)."""
    registry.register(ARTIFACT_RETENTION_SWEEP_TASK_NAME, make_retention_sweep_handler(sweeper))
