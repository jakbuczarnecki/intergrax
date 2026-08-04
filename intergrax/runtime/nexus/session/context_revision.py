# © Artur Czarnecki. All rights reserved.

"""Memory/Session-owned durable context revision manifests and CAS activation."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from enum import StrEnum

from intergrax.runtime.context_lifecycle import (
    ArtifactValidationStatus,
    ReusableArtifactStatus,
    compute_artifact_lookup_key_hash,
    OptimizationArtifactRepository,
)
from intergrax.runtime.token_optimization.durable_compaction_validation import (
    DurableCompactionValidationOutcome,
    DurableCompactionValidationStatus,
)


class SessionContextRevisionActivationStatus(StrEnum):
    ACTIVATED = "activated"
    STALE_CONTEXT_REVISION = "stale_context_revision"
    ALREADY_ACTIVATED = "already_activated"


class SessionContextRevisionActivationError(RuntimeError):
    """Safe activation failure; never embeds artifact payload or summary text."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _revision(value: object, field_name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{field_name} must be {'> 0' if positive else '>= 0'}")
    return value


def _aware(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class SessionContextRevision:
    """Immutable, append-only manifest; it never contains context content."""

    manifest_id: str
    tenant_id: str
    context_scope_id: str
    revision: int
    parent_revision: int
    artifact_id: str
    artifact_content_hash: str
    artifact_lookup_key_hash: str
    source_identity_hash: str
    lineage_reference: str
    creation_receipt_reference: str
    rollback_source_reference: str
    created_at: datetime
    created_by_operation_id: str
    prior_artifact_id: str | None = None
    active: bool = False
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.manifest_id, "manifest_id"),
            (self.tenant_id, "tenant_id"),
            (self.context_scope_id, "context_scope_id"),
            (self.artifact_id, "artifact_id"),
            (self.artifact_content_hash, "artifact_content_hash"),
            (self.artifact_lookup_key_hash, "artifact_lookup_key_hash"),
            (self.source_identity_hash, "source_identity_hash"),
            (self.lineage_reference, "lineage_reference"),
            (self.creation_receipt_reference, "creation_receipt_reference"),
            (self.rollback_source_reference, "rollback_source_reference"),
            (self.created_by_operation_id, "created_by_operation_id"),
        ):
            _text(value, name)
        _revision(self.revision, "revision", positive=True)
        _revision(self.parent_revision, "parent_revision")
        if self.parent_revision != self.revision - 1:
            raise ValueError("parent_revision must immediately precede revision")
        _aware(self.created_at, "created_at")
        if self.active is not False:
            raise ValueError("revision manifests are immutable and inactive")
        if self.raw_content_included is not False:
            raise ValueError("raw_content_included must be False")
        if self.prior_artifact_id is not None:
            _text(self.prior_artifact_id, "prior_artifact_id")


@dataclass(frozen=True, slots=True)
class ActiveContextRevisionPointer:
    """Durable pointer kept separately from immutable revision manifests."""

    tenant_id: str
    context_scope_id: str
    active_revision: int
    active_artifact_id: str | None
    updated_at: datetime
    state_version: int

    def __post_init__(self) -> None:
        _text(self.tenant_id, "tenant_id")
        _text(self.context_scope_id, "context_scope_id")
        _revision(self.active_revision, "active_revision")
        if self.active_artifact_id is not None:
            _text(self.active_artifact_id, "active_artifact_id")
        _aware(self.updated_at, "updated_at")
        _revision(self.state_version, "state_version")


@dataclass(frozen=True, slots=True)
class SessionContextRevisionActivationRequest:
    """Memory/Session activation input bound to one validated TOKEN-10E-3 outcome."""

    tenant_id: str
    context_scope_id: str
    operation_id: str
    outcome: DurableCompactionValidationOutcome
    expected_active_revision: int
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        _text(self.tenant_id, "tenant_id")
        _text(self.context_scope_id, "context_scope_id")
        _text(self.operation_id, "operation_id")
        if type(self.outcome) is not DurableCompactionValidationOutcome:
            raise ValueError("outcome must be DurableCompactionValidationOutcome")
        _revision(self.expected_active_revision, "expected_active_revision")
        if self.raw_content_included is not False:
            raise ValueError("raw_content_included must be False")


@dataclass(frozen=True, slots=True)
class SessionContextRevisionActivationResult:
    """Redaction-safe immutable activation receipt."""

    status: SessionContextRevisionActivationStatus
    tenant_id: str
    context_scope_id: str
    previous_revision: int
    active_revision: int
    revision_manifest_id: str
    artifact_id: str
    artifact_content_hash: str
    lineage_reference: str
    creation_receipt_reference: str
    rollback_source_reference: str
    operation_id: str
    activated_at: datetime
    idempotent_replay: bool = False
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, SessionContextRevisionActivationStatus):
            raise ValueError("status must be SessionContextRevisionActivationStatus")
        for value, name in (
            (self.tenant_id, "tenant_id"),
            (self.context_scope_id, "context_scope_id"),
            (self.revision_manifest_id, "revision_manifest_id"),
            (self.artifact_id, "artifact_id"),
            (self.artifact_content_hash, "artifact_content_hash"),
            (self.lineage_reference, "lineage_reference"),
            (self.creation_receipt_reference, "creation_receipt_reference"),
            (self.rollback_source_reference, "rollback_source_reference"),
            (self.operation_id, "operation_id"),
        ):
            _text(value, name)
        _revision(self.previous_revision, "previous_revision")
        _revision(self.active_revision, "active_revision")
        _aware(self.activated_at, "activated_at")
        if self.status is SessionContextRevisionActivationStatus.ACTIVATED:
            if self.active_revision != self.previous_revision + 1:
                raise ValueError("ACTIVATED result must advance exactly one revision")
        if self.status is SessionContextRevisionActivationStatus.STALE_CONTEXT_REVISION:
            if self.idempotent_replay:
                raise ValueError("stale result cannot be an idempotent replay")
        if self.raw_content_included is not False:
            raise ValueError("raw_content_included must be False")


class SQLiteSessionContextRevisionStore:
    """Durable Memory/Session revision store using the canonical SQLite backend."""

    def __init__(
        self,
        db_path: str,
        *,
        clock: Callable[[], datetime] | None = None,
        manifest_id_factory: Callable[[], str] | None = None,
    ) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._manifest_id_factory = manifest_id_factory or (lambda: str(uuid.uuid4()))
        self._closed = False
        self._initialize_schema()

    def get_active_pointer(
        self,
        *,
        tenant_id: str,
        context_scope_id: str,
    ) -> ActiveContextRevisionPointer:
        _text(tenant_id, "tenant_id")
        _text(context_scope_id, "context_scope_id")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT tenant_id, context_scope_id, active_revision,
                       active_artifact_id, updated_at, state_version
                FROM active_context_revision_pointers
                WHERE tenant_id = ? AND context_scope_id = ?
                """,
                (tenant_id, context_scope_id),
            ).fetchone()
            if row is None:
                return ActiveContextRevisionPointer(
                    tenant_id=tenant_id,
                    context_scope_id=context_scope_id,
                    active_revision=0,
                    active_artifact_id=None,
                    updated_at=self._now(),
                    state_version=0,
                )
            return self._pointer_from_row(row)

    def get_revision(
        self,
        *,
        tenant_id: str,
        context_scope_id: str,
        revision: int,
    ) -> SessionContextRevision | None:
        _revision(revision, "revision", positive=True)
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT * FROM session_context_revisions
                WHERE tenant_id = ? AND context_scope_id = ? AND revision = ?
                """,
                (tenant_id, context_scope_id, revision),
            ).fetchone()
            return self._revision_from_row(row) if row is not None else None

    def activate(
        self,
        *,
        tenant_id: str,
        context_scope_id: str,
        operation_id: str,
        expected_active_revision: int,
        artifact_id: str,
        artifact_content_hash: str,
        artifact_lookup_key_hash: str,
        lineage_reference: str,
        creation_receipt_reference: str,
        rollback_source_reference: str,
        source_identity_hash: str,
        prior_artifact_id: str | None,
        build_manifest: Callable[[int, int, datetime, str], SessionContextRevision],
    ) -> SessionContextRevisionActivationResult:
        with self._lock:
            self._ensure_open()
            self._begin()
            try:
                existing_row = self._connection.execute(
                    """
                    SELECT * FROM session_context_revisions
                    WHERE tenant_id = ? AND context_scope_id = ?
                      AND created_by_operation_id = ?
                    """,
                    (tenant_id, context_scope_id, operation_id),
                ).fetchone()
                if existing_row is not None:
                    existing = self._revision_from_row(existing_row)
                    if not self._same_operation(
                        existing,
                        expected_active_revision=expected_active_revision,
                        artifact_id=artifact_id,
                        artifact_content_hash=artifact_content_hash,
                        artifact_lookup_key_hash=artifact_lookup_key_hash,
                        lineage_reference=lineage_reference,
                        creation_receipt_reference=creation_receipt_reference,
                        rollback_source_reference=rollback_source_reference,
                        source_identity_hash=source_identity_hash,
                        prior_artifact_id=prior_artifact_id,
                    ):
                        raise SessionContextRevisionActivationError("OPERATION_ID_CONFLICT")
                    self._commit()
                    return self._result_from_revision(
                        existing,
                        status=SessionContextRevisionActivationStatus.ALREADY_ACTIVATED,
                        idempotent_replay=True,
                    )

                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO active_context_revision_pointers (
                        tenant_id, context_scope_id, active_revision,
                        active_artifact_id, updated_at, state_version
                    ) VALUES (?, ?, 0, NULL, ?, 0)
                    """,
                    (tenant_id, context_scope_id, self._now().isoformat()),
                )
                pointer_row = self._connection.execute(
                    """
                    SELECT tenant_id, context_scope_id, active_revision,
                           active_artifact_id, updated_at, state_version
                    FROM active_context_revision_pointers
                    WHERE tenant_id = ? AND context_scope_id = ?
                    """,
                    (tenant_id, context_scope_id),
                ).fetchone()
                if pointer_row is None:
                    raise SessionContextRevisionActivationError("POINTER_UNAVAILABLE")
                pointer = self._pointer_from_row(pointer_row)
                if pointer.active_revision != expected_active_revision:
                    self._commit()
                    return SessionContextRevisionActivationResult(
                        status=SessionContextRevisionActivationStatus.STALE_CONTEXT_REVISION,
                        tenant_id=tenant_id,
                        context_scope_id=context_scope_id,
                        previous_revision=pointer.active_revision,
                        active_revision=pointer.active_revision,
                        revision_manifest_id="none",
                        artifact_id=artifact_id,
                        artifact_content_hash=artifact_content_hash,
                        lineage_reference=lineage_reference,
                        creation_receipt_reference=creation_receipt_reference,
                        rollback_source_reference=rollback_source_reference,
                        operation_id=operation_id,
                        activated_at=self._now(),
                    )

                now = self._now()
                manifest = build_manifest(
                    pointer.active_revision,
                    pointer.active_revision + 1,
                    now,
                    self._new_manifest_id(),
                )
                if manifest.parent_revision != pointer.active_revision:
                    raise SessionContextRevisionActivationError("MANIFEST_PARENT_CONFLICT")
                self._insert_manifest(manifest)
                updated = self._connection.execute(
                    """
                    UPDATE active_context_revision_pointers
                    SET active_revision = ?,
                        active_artifact_id = ?,
                        updated_at = ?,
                        state_version = state_version + 1
                    WHERE tenant_id = ? AND context_scope_id = ?
                      AND active_revision = ?
                    """,
                    (
                        manifest.revision,
                        manifest.artifact_id,
                        now.isoformat(),
                        tenant_id,
                        context_scope_id,
                        expected_active_revision,
                    ),
                ).rowcount
                if updated != 1:
                    raise SessionContextRevisionActivationError("STALE_CONTEXT_REVISION")
                self._commit()
                return self._result_from_revision(
                    manifest,
                    status=SessionContextRevisionActivationStatus.ACTIVATED,
                    idempotent_replay=False,
                )
            except Exception:
                self._rollback()
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_context_revisions (
                    tenant_id TEXT NOT NULL,
                    context_scope_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    manifest_id TEXT NOT NULL,
                    parent_revision INTEGER NOT NULL,
                    artifact_id TEXT NOT NULL,
                    artifact_content_hash TEXT NOT NULL,
                    artifact_lookup_key_hash TEXT NOT NULL,
                    source_identity_hash TEXT NOT NULL,
                    lineage_reference TEXT NOT NULL,
                    creation_receipt_reference TEXT NOT NULL,
                    rollback_source_reference TEXT NOT NULL,
                    prior_artifact_id TEXT,
                    created_at TEXT NOT NULL,
                    created_by_operation_id TEXT NOT NULL,
                    active INTEGER NOT NULL CHECK (active = 0),
                    raw_content_included INTEGER NOT NULL CHECK (raw_content_included = 0),
                    PRIMARY KEY (tenant_id, context_scope_id, revision),
                    UNIQUE (tenant_id, context_scope_id, manifest_id),
                    UNIQUE (tenant_id, context_scope_id, created_by_operation_id)
                );
                CREATE TABLE IF NOT EXISTS active_context_revision_pointers (
                    tenant_id TEXT NOT NULL,
                    context_scope_id TEXT NOT NULL,
                    active_revision INTEGER NOT NULL,
                    active_artifact_id TEXT,
                    updated_at TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, context_scope_id)
                );
                """
            )
            self._connection.commit()

    def _begin(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._connection.commit()

    def _rollback(self) -> None:
        self._connection.rollback()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Session context revision store is closed")

    def _now(self) -> datetime:
        return _aware(self._clock(), "clock")

    def _new_manifest_id(self) -> str:
        return _text(self._manifest_id_factory(), "manifest_id_factory")

    @staticmethod
    def _same_operation(
        revision: SessionContextRevision,
        *,
        expected_active_revision: int,
        artifact_id: str,
        artifact_content_hash: str,
        artifact_lookup_key_hash: str,
        lineage_reference: str,
        creation_receipt_reference: str,
        rollback_source_reference: str,
        source_identity_hash: str,
        prior_artifact_id: str | None,
    ) -> bool:
        return (
            revision.parent_revision == expected_active_revision
            and revision.artifact_id == artifact_id
            and revision.artifact_content_hash == artifact_content_hash
            and revision.artifact_lookup_key_hash == artifact_lookup_key_hash
            and revision.lineage_reference == lineage_reference
            and revision.creation_receipt_reference == creation_receipt_reference
            and revision.rollback_source_reference == rollback_source_reference
            and revision.source_identity_hash == source_identity_hash
            and revision.prior_artifact_id == prior_artifact_id
        )

    def _insert_manifest(self, manifest: SessionContextRevision) -> None:
        self._connection.execute(
            """
            INSERT INTO session_context_revisions (
                tenant_id, context_scope_id, revision, manifest_id,
                parent_revision, artifact_id, artifact_content_hash,
                artifact_lookup_key_hash, source_identity_hash, lineage_reference,
                creation_receipt_reference, rollback_source_reference,
                prior_artifact_id, created_at, created_by_operation_id,
                active, raw_content_included
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (
                manifest.tenant_id,
                manifest.context_scope_id,
                manifest.revision,
                manifest.manifest_id,
                manifest.parent_revision,
                manifest.artifact_id,
                manifest.artifact_content_hash,
                manifest.artifact_lookup_key_hash,
                manifest.source_identity_hash,
                manifest.lineage_reference,
                manifest.creation_receipt_reference,
                manifest.rollback_source_reference,
                manifest.prior_artifact_id,
                manifest.created_at.isoformat(),
                manifest.created_by_operation_id,
            ),
        )

    @staticmethod
    def _pointer_from_row(row: tuple[object, ...]) -> ActiveContextRevisionPointer:
        return ActiveContextRevisionPointer(
            tenant_id=str(row[0]),
            context_scope_id=str(row[1]),
            active_revision=int(row[2]),
            active_artifact_id=str(row[3]) if row[3] is not None else None,
            updated_at=datetime.fromisoformat(str(row[4])),
            state_version=int(row[5]),
        )

    @staticmethod
    def _revision_from_row(row: tuple[object, ...]) -> SessionContextRevision:
        return SessionContextRevision(
            tenant_id=str(row[0]),
            context_scope_id=str(row[1]),
            revision=int(row[2]),
            manifest_id=str(row[3]),
            parent_revision=int(row[4]),
            artifact_id=str(row[5]),
            artifact_content_hash=str(row[6]),
            artifact_lookup_key_hash=str(row[7]),
            source_identity_hash=str(row[8]),
            lineage_reference=str(row[9]),
            creation_receipt_reference=str(row[10]),
            rollback_source_reference=str(row[11]),
            prior_artifact_id=str(row[12]) if row[12] is not None else None,
            created_at=datetime.fromisoformat(str(row[13])),
            created_by_operation_id=str(row[14]),
        )

    def _result_from_revision(
        self,
        revision: SessionContextRevision,
        *,
        status: SessionContextRevisionActivationStatus,
        idempotent_replay: bool,
    ) -> SessionContextRevisionActivationResult:
        return SessionContextRevisionActivationResult(
            status=status,
            tenant_id=revision.tenant_id,
            context_scope_id=revision.context_scope_id,
            previous_revision=revision.parent_revision,
            active_revision=revision.revision,
            revision_manifest_id=revision.manifest_id,
            artifact_id=revision.artifact_id,
            artifact_content_hash=revision.artifact_content_hash,
            lineage_reference=revision.lineage_reference,
            creation_receipt_reference=revision.creation_receipt_reference,
            rollback_source_reference=revision.rollback_source_reference,
            operation_id=revision.created_by_operation_id,
            activated_at=revision.created_at,
            idempotent_replay=idempotent_replay,
        )


class SessionContextRevisionActivationService:
    """Memory/Session owner of validated-artifact activation; no executor/model dependency."""

    def __init__(
        self,
        *,
        repository: OptimizationArtifactRepository,
        revision_store: SQLiteSessionContextRevisionStore,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(repository, OptimizationArtifactRepository):
            raise ValueError("repository must implement OptimizationArtifactRepository")
        self._repository = repository
        self._revision_store = revision_store
        self._clock = clock or (lambda: datetime.now(UTC))

    def activate(
        self,
        request: SessionContextRevisionActivationRequest,
    ) -> SessionContextRevisionActivationResult:
        if not isinstance(request, SessionContextRevisionActivationRequest):
            raise ValueError("request must be SessionContextRevisionActivationRequest")
        outcome = request.outcome
        self._validate_outcome(request)
        candidate = outcome.candidate
        reference = candidate.artifact_reference
        try:
            stored = self._repository.resolve(reference)
        except Exception as exc:
            raise SessionContextRevisionActivationError("ARTIFACT_RESOLVE_FAILED") from exc
        if stored is None:
            raise SessionContextRevisionActivationError("ARTIFACT_NOT_FOUND")
        metadata = stored.metadata
        if (
            metadata.status is not ReusableArtifactStatus.VALIDATED
            or metadata.validation.status is not ArtifactValidationStatus.PASSED
            or metadata.lookup_key.tenant_id != request.tenant_id
            or metadata.lookup_key.context_scope_id != request.context_scope_id
            or metadata.artifact_id != reference.artifact_id
            or metadata.artifact_content_hash != reference.artifact_content_hash
            or compute_artifact_lookup_key_hash(metadata.lookup_key)
            != reference.artifact_lookup_key_hash
        ):
            raise SessionContextRevisionActivationError("ARTIFACT_REVALIDATION_FAILED")

        rollback = outcome.rollback_metadata
        requirements = outcome.activation_requirements
        assert rollback is not None
        assert requirements is not None
        prior_artifact_id = (
            rollback.prior_artifact_reference.artifact_id
            if rollback.prior_artifact_reference is not None
            else None
        )

        def build_manifest(
            parent_revision: int,
            new_revision: int,
            created_at: datetime,
            manifest_id: str,
        ) -> SessionContextRevision:
            return SessionContextRevision(
                manifest_id=manifest_id,
                tenant_id=request.tenant_id,
                context_scope_id=request.context_scope_id,
                revision=new_revision,
                parent_revision=parent_revision,
                artifact_id=reference.artifact_id,
                artifact_content_hash=reference.artifact_content_hash,
                artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
                source_identity_hash=rollback.source_identity_hash,
                lineage_reference=requirements.lineage_reference,
                creation_receipt_reference=requirements.creation_receipt_reference,
                rollback_source_reference=rollback.rollback_source_reference,
                prior_artifact_id=prior_artifact_id,
                created_at=created_at,
                created_by_operation_id=request.operation_id,
            )

        return self._revision_store.activate(
            tenant_id=request.tenant_id,
            context_scope_id=request.context_scope_id,
            operation_id=request.operation_id,
            expected_active_revision=request.expected_active_revision,
            artifact_id=reference.artifact_id,
            artifact_content_hash=reference.artifact_content_hash,
            artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
            lineage_reference=requirements.lineage_reference,
            creation_receipt_reference=requirements.creation_receipt_reference,
            rollback_source_reference=rollback.rollback_source_reference,
            source_identity_hash=rollback.source_identity_hash,
            prior_artifact_id=prior_artifact_id,
            build_manifest=build_manifest,
        )

    @staticmethod
    def _validate_outcome(request: SessionContextRevisionActivationRequest) -> None:
        outcome = request.outcome
        if outcome.status is not DurableCompactionValidationStatus.PASSED:
            raise SessionContextRevisionActivationError("VALIDATION_OUTCOME_NOT_PASSED")
        if outcome.raw_content_included is not False:
            raise SessionContextRevisionActivationError("RAW_CONTENT_FORBIDDEN")
        if outcome.candidate.active is not False or outcome.candidate.raw_content_included is not False:
            raise SessionContextRevisionActivationError("CANDIDATE_MUST_BE_INACTIVE")
        if not outcome.receipt.validation_passed or outcome.receipt.raw_content_included is not False:
            raise SessionContextRevisionActivationError("RECEIPT_NOT_VALIDATED")
        if outcome.receipt.invalidated_prior_artifact is not False:
            raise SessionContextRevisionActivationError("PRIOR_ARTIFACT_INVALIDATION_FORBIDDEN")
        rollback = outcome.rollback_metadata
        requirements = outcome.activation_requirements
        if rollback is None or requirements is None:
            raise SessionContextRevisionActivationError("ACTIVATION_METADATA_MISSING")
        reference = outcome.candidate.artifact_reference
        expected = request.expected_active_revision
        expected_values = (
            outcome.receipt.expected_active_revision,
            rollback.expected_active_revision,
            requirements.expected_active_revision,
        )
        if any(value != expected for value in expected_values):
            raise SessionContextRevisionActivationError("EXPECTED_REVISION_MISMATCH")
        if (
            request.tenant_id != outcome.receipt.tenant_id
            or request.context_scope_id != outcome.receipt.context_scope_id
            or rollback.tenant_id != request.tenant_id
            or rollback.context_scope_id != request.context_scope_id
            or reference.tenant_id != request.tenant_id
        ):
            raise SessionContextRevisionActivationError("TENANT_SCOPE_MISMATCH")
        if (
            requirements.candidate_artifact_id != reference.artifact_id
            or requirements.validated_artifact_id != reference.artifact_id
            or outcome.receipt.artifact_id != reference.artifact_id
            or rollback.candidate_artifact_reference != reference
        ):
            raise SessionContextRevisionActivationError("ARTIFACT_ID_MISMATCH")
        if (
            candidate_hash := outcome.candidate.artifact_content_hash
        ) != reference.artifact_content_hash:
            raise SessionContextRevisionActivationError("ARTIFACT_CONTENT_HASH_MISMATCH")
        if (
            outcome.candidate.artifact_lookup_hash != reference.artifact_lookup_key_hash
            or outcome.receipt.artifact_lookup_hash != reference.artifact_lookup_key_hash
        ):
            raise SessionContextRevisionActivationError("ARTIFACT_LOOKUP_HASH_MISMATCH")
        if (
            outcome.receipt.artifact_content_hash != candidate_hash
            or rollback.candidate_artifact_content_hash != candidate_hash
            or outcome.candidate.source_identity_hash != rollback.source_identity_hash
            or outcome.receipt.source_identity_hash != rollback.source_identity_hash
            or requirements.rollback_source_reference != rollback.rollback_source_reference
            or requirements.creation_receipt_reference != outcome.receipt.receipt_id
        ):
            raise SessionContextRevisionActivationError("OUTCOME_REFERENCE_MISMATCH")
