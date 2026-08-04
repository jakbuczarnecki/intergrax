# © Artur Czarnecki. All rights reserved.

"""Durable SQLite adapter for the UCL optimization artifact repository."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompressionTarget,
    ArtifactCreationCoordinationStatus,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactSourceRange,
    ArtifactValidationStatus,
    ContextOptimizationReasonCode,
    OptimizationArtifactType,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
    ArtifactValidationSummary,
)
from intergrax.runtime.context_lifecycle.repository import (
    ArtifactCreationCoordinationResult,
    OptimizationArtifactReference,
    OptimizationArtifactRepositoryCapabilities,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
)
from intergrax.runtime.context_lifecycle.serialization import (
    artifact_lookup_key_to_canonical_dict,
    compute_artifact_lookup_key_hash,
)

_CLOSED_ERROR = "Optimization artifact repository is closed"
_ACTIVE_RESERVATION = "active"
_STORED_RESERVATION = "stored"
_RELEASED_RESERVATION = "released"
_EXPIRED_RESERVATION = "expired"


def _require_non_empty(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout_seconds must be finite and >= 0")
    return timeout


def _aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware datetime")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _decode_lookup_key(value: str) -> ArtifactLookupKey:
    payload = json.loads(value)
    compression = payload["compression_target"]
    target = ArtifactCompressionTarget(**compression)
    source_range_payload = payload.get("source_range")
    source_range = (
        ArtifactSourceRange(**source_range_payload)
        if source_range_payload is not None
        else None
    )
    return ArtifactLookupKey(
        tenant_id=payload["tenant_id"],
        context_scope_id=payload["context_scope_id"],
        artifact_type=OptimizationArtifactType(payload["artifact_type"]),
        source_content_hash=payload["source_content_hash"],
        strategy_id=payload["strategy_id"],
        strategy_version=payload["strategy_version"],
        policy_version=payload["policy_version"],
        validation_contract_version=payload["validation_contract_version"],
        compression_target=target,
        lossiness_profile=payload["lossiness_profile"],
        source_refs=tuple(payload.get("source_refs", ())),
        source_range=source_range,
        protected_region_policy_version=payload.get("protected_region_policy_version"),
        model_family=payload.get("model_family"),
        locale=payload.get("locale"),
    )


class SQLiteOptimizationArtifactRepository:
    """Process- and restart-shared implementation of the existing UCL port."""

    def __init__(
        self,
        db_path: str,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        reservation_id_factory: Callable[[], str] | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if clock is not None and not callable(clock):
            raise ValueError("clock must be callable when provided")
        if monotonic_clock is not None and not callable(monotonic_clock):
            raise ValueError("monotonic_clock must be callable when provided")
        if sleep is not None and not callable(sleep):
            raise ValueError("sleep must be callable when provided")
        if reservation_id_factory is not None and not callable(reservation_id_factory):
            raise ValueError("reservation_id_factory must be callable when provided")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be finite and > 0")

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
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._reservation_id_factory = reservation_id_factory or (lambda: str(uuid.uuid4()))
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._closed = False
        self._initialize_schema()

    @property
    def capabilities(self) -> OptimizationArtifactRepositoryCapabilities:
        return OptimizationArtifactRepositoryCapabilities(
            backend_id="sqlite",
            durable=True,
            shared_across_processes=True,
            supports_single_flight=True,
            supports_bounded_wait=True,
            reference_only=False,
        )

    def lookup(self, key: ArtifactLookupKey) -> StoredOptimizationArtifact | None:
        lookup_key = self._require_lookup_key(key)
        key_hash = compute_artifact_lookup_key_hash(lookup_key)
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT * FROM optimization_artifacts
                WHERE tenant_id = ? AND lookup_key_hash = ?
                  AND status = ? AND validation_status = ?
                """,
                (
                    lookup_key.tenant_id,
                    key_hash,
                    ReusableArtifactStatus.VALIDATED.value,
                    ArtifactValidationStatus.PASSED.value,
                ),
            ).fetchone()
            if row is None:
                return None
            artifact = self._row_to_artifact(row)
            if artifact.metadata.lookup_key != lookup_key:
                raise RuntimeError(
                    ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
                )
            return artifact

    def resolve(self, reference: OptimizationArtifactReference) -> StoredOptimizationArtifact | None:
        if not isinstance(reference, OptimizationArtifactReference):
            raise ValueError("reference must be OptimizationArtifactReference")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT * FROM optimization_artifacts
                WHERE tenant_id = ? AND artifact_id = ?
                """,
                (reference.tenant_id, reference.artifact_id),
            ).fetchone()
            if row is None:
                return None
            artifact = self._row_to_artifact(row)
            metadata = artifact.metadata
            if (
                compute_artifact_lookup_key_hash(metadata.lookup_key)
                != reference.artifact_lookup_key_hash
                or metadata.artifact_content_hash != reference.artifact_content_hash
                or metadata.lookup_key.artifact_type is not reference.artifact_type
            ):
                return None
            return artifact

    def try_acquire_creation_reservation(
        self,
        key: ArtifactLookupKey,
        *,
        owner_operation_id: str,
        lease_seconds: int,
    ) -> ArtifactCreationCoordinationResult:
        lookup_key = self._require_lookup_key(key)
        owner = _require_non_empty(owner_operation_id, "owner_operation_id")
        lease = _require_positive_int(lease_seconds, "lease_seconds")
        key_hash = compute_artifact_lookup_key_hash(lookup_key)
        with self._lock:
            self._ensure_open()
            self._begin()
            try:
                state_version = self._next_state_version(lookup_key.tenant_id, key_hash)
                artifact = self._lookup_in_transaction(lookup_key, key_hash)
                if artifact is not None:
                    reference = build_optimization_artifact_reference(artifact)
                    self._commit()
                    return ArtifactCreationCoordinationResult(
                        status=ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE,
                        artifact_lookup_key_hash=key_hash,
                        state_version=state_version,
                        artifact_reference=reference,
                    )

                row = self._active_reservation_row(lookup_key.tenant_id, key_hash)
                if row is not None:
                    reservation = self._reservation_from_row(row)
                    if reservation.lease_deadline <= self._now():
                        next_version = state_version + 1
                        self._connection.execute(
                            """
                            UPDATE optimization_artifact_reservations
                            SET state = ?, state_version = ?
                            WHERE reservation_id = ?
                            """,
                            (_EXPIRED_RESERVATION, next_version, reservation.reservation_id),
                        )
                        self._commit()
                        return ArtifactCreationCoordinationResult(
                            status=ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED,
                            artifact_lookup_key_hash=key_hash,
                            state_version=next_version,
                            reservation=reservation,
                            reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED,
                        )
                    self._commit()
                    if reservation.owner_operation_id == owner:
                        return ArtifactCreationCoordinationResult(
                            status=ArtifactCreationCoordinationStatus.ACQUIRED,
                            artifact_lookup_key_hash=key_hash,
                            state_version=int(row[7]),
                            reservation=reservation,
                        )
                    return ArtifactCreationCoordinationResult(
                        status=ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS,
                        artifact_lookup_key_hash=key_hash,
                        state_version=int(row[7]),
                        reservation=reservation,
                        reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_IN_PROGRESS,
                    )

                now = self._now()
                reservation = ArtifactCreationReservation(
                    reservation_id=self._new_reservation_id(),
                    artifact_lookup_key_hash=key_hash,
                    tenant_id=lookup_key.tenant_id,
                    owner_operation_id=owner,
                    acquired_at=now,
                    lease_deadline=now + timedelta(seconds=lease),
                )
                next_version = state_version + 1
                self._connection.execute(
                    """
                    INSERT INTO optimization_artifact_reservations (
                        reservation_id, tenant_id, lookup_key_hash, owner_operation_id,
                        acquired_at, lease_deadline, state, state_version, reason_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        reservation.reservation_id,
                        reservation.tenant_id,
                        reservation.artifact_lookup_key_hash,
                        reservation.owner_operation_id,
                        reservation.acquired_at.isoformat(),
                        reservation.lease_deadline.isoformat(),
                        _ACTIVE_RESERVATION,
                        next_version,
                    ),
                )
                self._commit()
                return ArtifactCreationCoordinationResult(
                    status=ArtifactCreationCoordinationStatus.ACQUIRED,
                    artifact_lookup_key_hash=key_hash,
                    state_version=next_version,
                    reservation=reservation,
                )
            except Exception:
                self._rollback()
                raise

    def store_validated_artifact(
        self,
        *,
        reservation: ArtifactCreationReservation,
        artifact: StoredOptimizationArtifact,
    ) -> OptimizationArtifactReference:
        if not isinstance(reservation, ArtifactCreationReservation):
            raise ValueError("reservation must be ArtifactCreationReservation")
        if not isinstance(artifact, StoredOptimizationArtifact):
            raise ValueError("artifact must be StoredOptimizationArtifact")
        metadata = artifact.metadata
        if metadata.status is not ReusableArtifactStatus.VALIDATED:
            raise ValueError("artifact metadata.status must be VALIDATED")
        if metadata.validation.status is not ArtifactValidationStatus.PASSED:
            raise ValueError("artifact validation.status must be PASSED")
        lookup_key = metadata.lookup_key
        key_hash = compute_artifact_lookup_key_hash(lookup_key)
        if reservation.tenant_id != lookup_key.tenant_id:
            raise ValueError("reservation tenant_id must match artifact lookup_key tenant_id")
        if reservation.artifact_lookup_key_hash != key_hash:
            raise ValueError("reservation artifact_lookup_key_hash must match lookup key hash")

        with self._lock:
            self._ensure_open()
            self._begin()
            try:
                current = self._active_reservation_row(
                    reservation.tenant_id,
                    reservation.artifact_lookup_key_hash,
                )
                if current is None:
                    existing = self._artifact_by_id(
                        reservation.tenant_id,
                        metadata.artifact_id,
                    )
                    if existing is not None and self._same_artifact(existing, artifact, key_hash):
                        self._commit()
                        return build_optimization_artifact_reference(existing)
                    raise RuntimeError(
                        ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
                    )
                current_reservation = self._reservation_from_row(current)
                if current_reservation.reservation_id != reservation.reservation_id:
                    raise ValueError("reservation_id does not match active reservation")
                if current_reservation.owner_operation_id != reservation.owner_operation_id:
                    raise ValueError("owner_operation_id does not match active reservation")
                if current_reservation.lease_deadline <= self._now():
                    next_version = self._next_state_version(
                        lookup_key.tenant_id,
                        key_hash,
                    ) + 1
                    self._connection.execute(
                        """
                        UPDATE optimization_artifact_reservations
                        SET state = ?, state_version = ?, reason_code = ?
                        WHERE reservation_id = ?
                        """,
                        (
                            _EXPIRED_RESERVATION,
                            next_version,
                            ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED.value,
                            reservation.reservation_id,
                        ),
                    )
                    self._commit()
                    raise RuntimeError(
                        ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED.value
                    )

                existing_active = self._lookup_in_transaction(lookup_key, key_hash)
                if existing_active is not None:
                    if self._same_artifact(existing_active, artifact, key_hash):
                        next_version = self._next_state_version(lookup_key.tenant_id, key_hash) + 1
                        self._mark_reservation_stored(reservation.reservation_id, next_version)
                        self._commit()
                        return build_optimization_artifact_reference(existing_active)
                    raise RuntimeError(
                        ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
                    )

                state_version = self._next_state_version(lookup_key.tenant_id, key_hash) + 1
                self._connection.execute(
                    """
                    INSERT INTO optimization_artifacts (
                        tenant_id, artifact_id, lookup_key_hash, lookup_key_json,
                        artifact_content_hash, payload, media_type, encoding, status,
                        validation_status, validation_contract_version, validated_at,
                        validation_reason_codes_json, validation_safe_metadata_json,
                        created_at, created_by_executor, invalidation_reason,
                        supersedes_artifact_id, receipt_ref, safe_metadata_json, state_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._artifact_values(artifact, key_hash, state_version),
                )
                self._mark_reservation_stored(reservation.reservation_id, state_version)
                self._commit()
                return build_optimization_artifact_reference(artifact)
            except Exception:
                self._rollback()
                raise

    def release_creation_reservation(
        self,
        *,
        reservation: ArtifactCreationReservation,
        reason_code: ContextOptimizationReasonCode | None = None,
    ) -> bool:
        if not isinstance(reservation, ArtifactCreationReservation):
            raise ValueError("reservation must be ArtifactCreationReservation")
        if reason_code is not None and not isinstance(reason_code, ContextOptimizationReasonCode):
            raise ValueError("reason_code must be ContextOptimizationReasonCode when provided")
        with self._lock:
            self._ensure_open()
            self._begin()
            try:
                row = self._connection.execute(
                    """
                    SELECT * FROM optimization_artifact_reservations
                    WHERE reservation_id = ? AND tenant_id = ? AND lookup_key_hash = ?
                    """,
                    (
                        reservation.reservation_id,
                        reservation.tenant_id,
                        reservation.artifact_lookup_key_hash,
                    ),
                ).fetchone()
                if row is None:
                    self._commit()
                    return False
                current = self._reservation_from_row(row)
                if (
                    current.owner_operation_id != reservation.owner_operation_id
                    or row[6] != _ACTIVE_RESERVATION
                ):
                    self._commit()
                    return False
                next_version = self._next_state_version(
                    reservation.tenant_id,
                    reservation.artifact_lookup_key_hash,
                ) + 1
                self._connection.execute(
                    """
                    UPDATE optimization_artifact_reservations
                    SET state = ?, state_version = ?, reason_code = ?
                    WHERE reservation_id = ?
                    """,
                    (
                        _RELEASED_RESERVATION,
                        next_version,
                        reason_code.value if reason_code is not None else None,
                        reservation.reservation_id,
                    ),
                )
                self._commit()
                return True
            except Exception:
                self._rollback()
                raise

    def wait_for_artifact_or_reservation_change(
        self,
        key: ArtifactLookupKey,
        *,
        observed_state_version: int,
        timeout_seconds: float,
    ) -> bool:
        lookup_key = self._require_lookup_key(key)
        observed = _require_non_negative_int(observed_state_version, "observed_state_version")
        timeout = _require_timeout(timeout_seconds)
        started = self._monotonic_now()
        while True:
            self._expire_if_needed(lookup_key)
            with self._lock:
                self._ensure_open()
                changed = self._state_version(
                    lookup_key.tenant_id,
                    compute_artifact_lookup_key_hash(lookup_key),
                ) > observed
            if changed:
                return True
            elapsed = self._monotonic_now() - started
            remaining = timeout - elapsed
            if remaining <= 0:
                return False
            self._sleep(min(self._poll_interval_seconds, remaining))

    def invalidate_artifact(
        self,
        reference: OptimizationArtifactReference,
        *,
        reason: str,
    ) -> StoredOptimizationArtifact | None:
        return self._transition_artifact(reference, ReusableArtifactStatus.INVALIDATED, reason)

    def retire_artifact(
        self,
        reference: OptimizationArtifactReference,
        *,
        reason: str,
    ) -> StoredOptimizationArtifact | None:
        return self._transition_artifact(reference, ReusableArtifactStatus.RETIRED, reason)

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
                CREATE TABLE IF NOT EXISTS optimization_artifacts (
                    tenant_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    lookup_key_hash TEXT NOT NULL,
                    lookup_key_json TEXT NOT NULL,
                    artifact_content_hash TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    media_type TEXT NOT NULL,
                    encoding TEXT,
                    status TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    validation_contract_version TEXT NOT NULL,
                    validated_at TEXT NOT NULL,
                    validation_reason_codes_json TEXT NOT NULL,
                    validation_safe_metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by_executor TEXT NOT NULL,
                    invalidation_reason TEXT,
                    supersedes_artifact_id TEXT,
                    receipt_ref TEXT,
                    safe_metadata_json TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, artifact_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS uq_optimization_artifact_active_lookup
                    ON optimization_artifacts (tenant_id, lookup_key_hash)
                    WHERE status = 'validated';
                CREATE TABLE IF NOT EXISTS optimization_artifact_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    lookup_key_hash TEXT NOT NULL,
                    owner_operation_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    lease_deadline TEXT NOT NULL,
                    state TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    reason_code TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS uq_optimization_artifact_active_reservation
                    ON optimization_artifact_reservations (tenant_id, lookup_key_hash)
                    WHERE state = 'active';
                """
            )
            self._connection.commit()

    def _initialize_transaction(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    _begin = _initialize_transaction

    def _commit(self) -> None:
        self._connection.commit()

    def _rollback(self) -> None:
        self._connection.rollback()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(_CLOSED_ERROR)

    @staticmethod
    def _require_lookup_key(key: object) -> ArtifactLookupKey:
        if not isinstance(key, ArtifactLookupKey):
            raise ValueError("key must be ArtifactLookupKey")
        return key

    def _now(self) -> datetime:
        return _aware_datetime(self._clock(), "clock")

    def _monotonic_now(self) -> float:
        value = self._monotonic_clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("monotonic_clock must return a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("monotonic_clock must return a finite number")
        return value

    def _new_reservation_id(self) -> str:
        value = _require_non_empty(self._reservation_id_factory(), "reservation_id_factory")
        return value

    def _state_version(self, tenant_id: str, key_hash: str) -> int:
        row = self._connection.execute(
            """
            SELECT MAX(state_version) FROM (
                SELECT state_version FROM optimization_artifacts
                WHERE tenant_id = ? AND lookup_key_hash = ?
                UNION ALL
                SELECT state_version FROM optimization_artifact_reservations
                WHERE tenant_id = ? AND lookup_key_hash = ?
            )
            """,
            (tenant_id, key_hash, tenant_id, key_hash),
        ).fetchone()
        return int(row[0] or 0)

    def _next_state_version(self, tenant_id: str, key_hash: str) -> int:
        return self._state_version(tenant_id, key_hash)

    def _active_reservation_row(self, tenant_id: str, key_hash: str) -> tuple[Any, ...] | None:
        return self._connection.execute(
            """
            SELECT * FROM optimization_artifact_reservations
            WHERE tenant_id = ? AND lookup_key_hash = ? AND state = ?
            """,
            (tenant_id, key_hash, _ACTIVE_RESERVATION),
        ).fetchone()

    def _lookup_in_transaction(
        self,
        key: ArtifactLookupKey,
        key_hash: str,
    ) -> StoredOptimizationArtifact | None:
        row = self._connection.execute(
            """
            SELECT * FROM optimization_artifacts
            WHERE tenant_id = ? AND lookup_key_hash = ?
              AND status = ? AND validation_status = ?
            """,
            (
                key.tenant_id,
                key_hash,
                ReusableArtifactStatus.VALIDATED.value,
                ArtifactValidationStatus.PASSED.value,
            ),
        ).fetchone()
        if row is None:
            return None
        artifact = self._row_to_artifact(row)
        if artifact.metadata.lookup_key != key:
            raise RuntimeError(
                ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
            )
        return artifact

    def _artifact_by_id(self, tenant_id: str, artifact_id: str) -> StoredOptimizationArtifact | None:
        row = self._connection.execute(
            """
            SELECT * FROM optimization_artifacts
            WHERE tenant_id = ? AND artifact_id = ?
            """,
            (tenant_id, artifact_id),
        ).fetchone()
        return self._row_to_artifact(row) if row is not None else None

    @staticmethod
    def _same_artifact(
        left: StoredOptimizationArtifact,
        right: StoredOptimizationArtifact,
        key_hash: str,
    ) -> bool:
        return (
            left.metadata.artifact_id == right.metadata.artifact_id
            and left.metadata.status is ReusableArtifactStatus.VALIDATED
            and left.metadata.validation.status is ArtifactValidationStatus.PASSED
            and compute_artifact_lookup_key_hash(left.metadata.lookup_key) == key_hash
            and left.metadata.artifact_content_hash == right.metadata.artifact_content_hash
            and left.payload == right.payload
        )

    def _mark_reservation_stored(self, reservation_id: str, state_version: int) -> None:
        self._connection.execute(
            """
            UPDATE optimization_artifact_reservations
            SET state = ?, state_version = ?
            WHERE reservation_id = ? AND state = ?
            """,
            (_STORED_RESERVATION, state_version, reservation_id, _ACTIVE_RESERVATION),
        )

    def _expire_if_needed(self, key: ArtifactLookupKey) -> None:
        key_hash = compute_artifact_lookup_key_hash(key)
        with self._lock:
            self._ensure_open()
            self._begin()
            try:
                row = self._active_reservation_row(key.tenant_id, key_hash)
                if row is None:
                    self._commit()
                    return
                reservation = self._reservation_from_row(row)
                if reservation.lease_deadline > self._now():
                    self._commit()
                    return
                next_version = self._next_state_version(key.tenant_id, key_hash) + 1
                self._connection.execute(
                    """
                    UPDATE optimization_artifact_reservations
                    SET state = ?, state_version = ?, reason_code = ?
                    WHERE reservation_id = ?
                    """,
                    (
                        _EXPIRED_RESERVATION,
                        next_version,
                        ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED.value,
                        reservation.reservation_id,
                    ),
                )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def _transition_artifact(
        self,
        reference: OptimizationArtifactReference,
        status: ReusableArtifactStatus,
        reason: str,
    ) -> StoredOptimizationArtifact | None:
        if not isinstance(reference, OptimizationArtifactReference):
            raise ValueError("reference must be OptimizationArtifactReference")
        reason = _require_non_empty(reason, "reason")
        with self._lock:
            self._ensure_open()
            self._begin()
            try:
                artifact = self._artifact_by_id(reference.tenant_id, reference.artifact_id)
                if artifact is None:
                    self._commit()
                    return None
                if (
                    compute_artifact_lookup_key_hash(artifact.metadata.lookup_key)
                    != reference.artifact_lookup_key_hash
                    or artifact.metadata.artifact_content_hash != reference.artifact_content_hash
                    or artifact.metadata.lookup_key.artifact_type is not reference.artifact_type
                ):
                    self._commit()
                    return None
                state_version = self._next_state_version(
                    reference.tenant_id,
                    reference.artifact_lookup_key_hash,
                ) + 1
                self._connection.execute(
                    """
                    UPDATE optimization_artifacts
                    SET status = ?, invalidation_reason = ?, state_version = ?
                    WHERE tenant_id = ? AND artifact_id = ?
                    """,
                    (
                        status.value,
                        reason,
                        state_version,
                        reference.tenant_id,
                        reference.artifact_id,
                    ),
                )
                updated = self._artifact_by_id(reference.tenant_id, reference.artifact_id)
                self._commit()
                return updated
            except Exception:
                self._rollback()
                raise

    def _artifact_values(
        self,
        artifact: StoredOptimizationArtifact,
        key_hash: str,
        state_version: int,
    ) -> tuple[Any, ...]:
        metadata = artifact.metadata
        validation = metadata.validation
        return (
            metadata.lookup_key.tenant_id,
            metadata.artifact_id,
            key_hash,
            json.dumps(
                artifact_lookup_key_to_canonical_dict(metadata.lookup_key),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            metadata.artifact_content_hash,
            sqlite3.Binary(artifact.payload),
            artifact.media_type,
            artifact.encoding,
            metadata.status.value,
            validation.status.value,
            validation.validation_contract_version,
            validation.validated_at.isoformat(),
            json.dumps(list(validation.reason_codes)),
            json.dumps(_json_safe(validation.safe_metadata), sort_keys=True),
            metadata.created_at.isoformat(),
            metadata.created_by_executor,
            metadata.invalidation_reason,
            metadata.supersedes_artifact_id,
            metadata.receipt_ref,
            json.dumps(_json_safe(metadata.safe_metadata), sort_keys=True),
            state_version,
        )

    def _row_to_artifact(self, row: tuple[Any, ...]) -> StoredOptimizationArtifact:
        lookup_key = _decode_lookup_key(str(row[3]))
        validation = ArtifactValidationSummary(
            status=ArtifactValidationStatus(str(row[9])),
            validation_contract_version=str(row[10]),
            validated_at=datetime.fromisoformat(str(row[11])),
            reason_codes=tuple(json.loads(str(row[12]))),
            safe_metadata=json.loads(str(row[13])),
        )
        metadata = ReusableOptimizationArtifact(
            artifact_id=str(row[1]),
            lookup_key=lookup_key,
            artifact_content_hash=str(row[4]),
            created_at=datetime.fromisoformat(str(row[14])),
            created_by_executor=str(row[15]),
            validation=validation,
            status=ReusableArtifactStatus(str(row[8])),
            invalidation_reason=str(row[16]) if row[16] is not None else None,
            supersedes_artifact_id=str(row[17]) if row[17] is not None else None,
            receipt_ref=str(row[18]) if row[18] is not None else None,
            safe_metadata=json.loads(str(row[19])),
        )
        return StoredOptimizationArtifact(
            metadata=metadata,
            payload=bytes(row[5]),
            media_type=str(row[6]),
            encoding=str(row[7]) if row[7] is not None else None,
        )

    @staticmethod
    def _reservation_from_row(row: tuple[Any, ...]) -> ArtifactCreationReservation:
        return ArtifactCreationReservation(
            reservation_id=str(row[0]),
            tenant_id=str(row[1]),
            artifact_lookup_key_hash=str(row[2]),
            owner_operation_id=str(row[3]),
            acquired_at=datetime.fromisoformat(str(row[4])),
            lease_deadline=datetime.fromisoformat(str(row[5])),
        )
