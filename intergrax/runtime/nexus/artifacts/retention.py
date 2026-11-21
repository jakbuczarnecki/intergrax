# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Retention class, TTL and the expiry sweep (PDC-D8-R008, R009, PDC-D6-R022).

The sweep is a handler on the existing canonical background registry
(``TaskExecutionRegistry``), run by ``execute_logical_task`` like any other background work. This
module adds no scheduler and no registry. Audit evidence and artifacts under legal hold are never
deleted by TTL: the sweeper skips them and the store refuses them as well.

EX-D8-01: ``produce_audit_evidence_artifact`` is the producer of ``AUDIT_EVIDENCE`` artifacts and
``require_retention_composition`` is the fail-closed composition gate a production root calls (the
store, the audit recorder and the sweep registration must all be present). No shipped root calls the
gate yet; wiring it into the worker root is outside this module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel

from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.artifact_ref import ArtifactClass, ArtifactRetentionClass
from intergrax.contracts.task_artifacts import ArtifactRetentionPolicy
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.queueing.worker.registry import BackgroundTaskHandler, TaskExecutionRegistry
from intergrax.runtime.agent_governance.audit import GovernanceAuditRecorder
from intergrax.runtime.nexus.artifacts.governed_access import (
    CAPABILITY_DELETE,
    ArtifactAccessContext,
    ArtifactOperationAuditor,
)
from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.store_base import (
    ArtifactDeletionRefused,
    ArtifactStore,
    ArtifactVersionChanged,
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


class ArtifactSweepAuditRequired(RuntimeError):
    """A sweep deletes data; without an audit recorder it is refused unless explicitly allowed."""


@dataclass(frozen=True)
class RetentionSweepReport:
    tenant_id: str
    examined: int
    deleted_ids: tuple[str, ...]
    protected_ids: tuple[str, ...]
    #: Selected as expired but changed (a newer version appeared) before deletion; left untouched.
    changed_ids: tuple[str, ...] = ()


class ArtifactRetentionSweeper:
    """Deletes expired, unprotected artifacts of one tenant; refuses evidence and legal hold."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        clock: Callable[[], datetime] | None = None,
        audit: GovernanceAuditRecorder | None = None,
        allow_unaudited_deletion: bool = False,
    ) -> None:
        self._store = store
        self._clock = clock or SystemTimeProvider.utc_now
        self._auditor = ArtifactOperationAuditor(audit) if audit is not None else None
        self._allow_unaudited = allow_unaudited_deletion

    def sweep(
        self,
        tenant_id: str,
        *,
        context: ArtifactAccessContext | None = None,
        limit: int = DEFAULT_SWEEP_LIMIT,
    ) -> RetentionSweepReport:
        if limit < 1:
            raise ValueError("sweep limit must be at least 1")
        if self._auditor is None and not self._allow_unaudited:
            raise ArtifactSweepAuditRequired(
                "an artifact retention sweep deletes data and needs an audit recorder "
                "(or allow_unaudited_deletion=True)"
            )
        if context is not None and context.tenant_id != tenant_id:
            raise ValueError("sweep context tenant differs from the swept tenant")
        if self._auditor is not None and context is None:
            raise ValueError("a sweep with an audit recorder needs an access context")
        now = self._clock()
        candidates = self._store.list_expired(tenant_id, now, limit=limit)
        deleted: list[str] = []
        protected: list[str] = []
        changed: list[str] = []
        for artifact in candidates:
            history_classes = [a.retention_class for a in self._store.list_versions(tenant_id, artifact.artifact_id)]
            reason = protection_reason(
                history_classes,
                legal_hold=self._store.is_under_legal_hold(tenant_id, artifact.artifact_id),
            )
            if reason is None:
                try:
                    self._store.delete_artifact(tenant_id, artifact.artifact_id, expected_version=artifact.version)
                except ArtifactDeletionRefused as exc:  # raced with a hold or evidence marking
                    reason = exc.reason
                except ArtifactVersionChanged:  # a newer version was written after the selection
                    changed.append(artifact.artifact_id)
                    self._record(
                        context,
                        artifact.artifact_id,
                        ToolAuthorizationDecisionState.DENY,
                        "version_changed",
                        "newer_version_written_after_selection;ttl_deletion_skipped",
                        now,
                    )
                    continue
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
            changed_ids=tuple(changed),
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
        try:
            report = sweeper.sweep(execution_identity.tenant_id, context=context, limit=limit)
        except ArtifactSweepAuditRequired:
            return ToolExecutionResult.fail(
                "audit_recorder_required", "the retention sweep needs an audit recorder and was not run"
            )
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


# ---------------------------------------------------------------------------
# EX-D8-01 - AUDIT_EVIDENCE producer and the retention composition gate
# ---------------------------------------------------------------------------


class RetentionCompositionError(ValueError):
    """A production composition lacks a retention input; raised at composition, never at first use."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        super().__init__("production retention composition is incomplete; missing: " + ", ".join(missing))
        self.missing = missing


def produce_audit_evidence_artifact(
    store: ArtifactStore,
    *,
    tenant_id: str,
    run_id: str,
    artifact_id: str,
    evidence_kind: str,
    data: bytes,
    mime_type: str = "application/json",
    step_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Artifact:
    """Write one audit evidence artifact (class ``AUDIT_EVIDENCE``: never expires, never deletable)."""
    for name, value in (
        ("tenant_id", tenant_id),
        ("run_id", run_id),
        ("artifact_id", artifact_id),
        ("evidence_kind", evidence_kind),
    ):
        if not value.strip():
            raise ValueError(f"audit evidence requires a non-blank {name}")
    now = (clock or SystemTimeProvider.utc_now)()
    artifact = Artifact(
        tenant_id=tenant_id.strip(),
        artifact_id=artifact_id.strip(),
        run_id=run_id.strip(),
        step_id=step_id,
        kind=evidence_kind.strip(),
        mime_type=mime_type,
        created_at_utc=now,
        data=data,
        size_bytes=len(data),
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
        retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
        expires_at_utc=resolve_expiry(now, retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE),
    )
    return store.put(artifact)


def _sweep_registered(registry: TaskExecutionRegistry) -> bool:
    try:
        registry.get_handler(ARTIFACT_RETENTION_SWEEP_TASK_NAME)
    except ValueError:
        return False
    return True


def require_retention_composition(
    *,
    production_mode: bool,
    artifact_store: ArtifactStore | None,
    audit: GovernanceAuditRecorder | None,
    sweep_registry: TaskExecutionRegistry | None,
) -> None:
    """Fail closed when a production root lacks the store, the audit recorder or the sweep registration.

    A non-production root may compose any subset (lab and unit hosts). For production the sweep
    handler must be registered on ``sweep_registry`` under ``ARTIFACT_RETENTION_SWEEP_TASK_NAME``.
    """
    if not production_mode:
        return
    missing: list[str] = []
    if artifact_store is None:
        missing.append("artifact_store")
    if audit is None:
        missing.append("audit_recorder")
    if sweep_registry is None or not _sweep_registered(sweep_registry):
        missing.append("retention_sweep_registration")
    if missing:
        raise RetentionCompositionError(tuple(missing))
