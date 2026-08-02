# © Artur Czarnecki. All rights reserved.

"""In-memory reference optimization artifact repository (CTX-UCL-2)."""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TypeAlias

from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCreationCoordinationStatus,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ContextOptimizationReasonCode,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
)
from intergrax.runtime.context_lifecycle.repository import (
    ArtifactCreationCoordinationResult,
    OptimizationArtifactReference,
    OptimizationArtifactRepositoryCapabilities,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
)
from intergrax.runtime.context_lifecycle.serialization import compute_artifact_lookup_key_hash

TenantKeyHash: TypeAlias = tuple[str, str]
TenantArtifactId: TypeAlias = tuple[str, str]

_CLOSED_ERROR = "Optimization artifact repository is closed"


def _require_lookup_key(key: object) -> ArtifactLookupKey:
    if not isinstance(key, ArtifactLookupKey):
        raise ValueError("key must be ArtifactLookupKey")
    return key


def _require_reference(reference: object) -> OptimizationArtifactReference:
    if not isinstance(reference, OptimizationArtifactReference):
        raise ValueError("reference must be OptimizationArtifactReference")
    return reference


def _require_reservation(reservation: object) -> ArtifactCreationReservation:
    if not isinstance(reservation, ArtifactCreationReservation):
        raise ValueError("reservation must be ArtifactCreationReservation")
    return reservation


def _require_stored_artifact(artifact: object) -> StoredOptimizationArtifact:
    if not isinstance(artifact, StoredOptimizationArtifact):
        raise ValueError("artifact must be StoredOptimizationArtifact")
    return artifact


def _require_non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _require_timeout_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout_seconds must be finite and >= 0")
    return timeout


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _default_reservation_id_factory() -> str:
    return str(uuid.uuid4())


class InMemoryOptimizationArtifactRepository:
    """Process-local reference repository for tests and local development."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        reservation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._clock = clock or _default_clock
        self._reservation_id_factory = reservation_id_factory or _default_reservation_id_factory
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closed = False

        self._active_by_key: dict[TenantKeyHash, StoredOptimizationArtifact] = {}
        self._artifacts_by_id: dict[TenantArtifactId, StoredOptimizationArtifact] = {}
        self._canonical_keys: dict[TenantKeyHash, ArtifactLookupKey] = {}
        self._reservations: dict[TenantKeyHash, ArtifactCreationReservation] = {}
        self._state_versions: dict[TenantKeyHash, int] = {}

        self._validate_clock()
        self._validate_reservation_id_factory()

    @property
    def capabilities(self) -> OptimizationArtifactRepositoryCapabilities:
        return OptimizationArtifactRepositoryCapabilities(
            backend_id="in_memory",
            durable=False,
            shared_across_processes=False,
            supports_single_flight=True,
            supports_bounded_wait=True,
            reference_only=True,
        )

    def lookup(self, key: ArtifactLookupKey) -> StoredOptimizationArtifact | None:
        lookup_key = _require_lookup_key(key)
        with self._lock:
            self._ensure_open()
            return self._lookup_eligible_locked(lookup_key)

    def resolve(self, reference: OptimizationArtifactReference) -> StoredOptimizationArtifact | None:
        ref = _require_reference(reference)
        with self._lock:
            self._ensure_open()
            return self._resolve_locked(ref)

    def try_acquire_creation_reservation(
        self,
        key: ArtifactLookupKey,
        *,
        owner_operation_id: str,
        lease_seconds: int,
    ) -> ArtifactCreationCoordinationResult:
        lookup_key = _require_lookup_key(key)
        owner = _require_non_empty(owner_operation_id, "owner_operation_id")
        lease = _require_positive_int(lease_seconds, "lease_seconds")
        key_hash = compute_artifact_lookup_key_hash(lookup_key)
        state_key = (lookup_key.tenant_id, key_hash)

        with self._lock:
            self._ensure_open()
            state_version = self._state_version_locked(state_key)

            existing_artifact = self._lookup_eligible_locked(lookup_key)
            if existing_artifact is not None:
                reference = build_optimization_artifact_reference(existing_artifact)
                return ArtifactCreationCoordinationResult(
                    status=ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE,
                    artifact_lookup_key_hash=key_hash,
                    state_version=state_version,
                    artifact_reference=reference,
                )

            reservation = self._reservations.get(state_key)
            if reservation is not None:
                now = self._clock()
                if reservation.lease_deadline <= now:
                    expired = reservation
                    self._remove_reservation_locked(state_key)
                    self._increment_state_version_locked(state_key)
                    self._condition.notify_all()
                    return ArtifactCreationCoordinationResult(
                        status=ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED,
                        artifact_lookup_key_hash=key_hash,
                        state_version=self._state_version_locked(state_key),
                        reservation=expired,
                        reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED,
                    )

                if reservation.owner_operation_id == owner:
                    return ArtifactCreationCoordinationResult(
                        status=ArtifactCreationCoordinationStatus.ACQUIRED,
                        artifact_lookup_key_hash=key_hash,
                        state_version=state_version,
                        reservation=reservation,
                    )

                return ArtifactCreationCoordinationResult(
                    status=ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS,
                    artifact_lookup_key_hash=key_hash,
                    state_version=state_version,
                    reservation=reservation,
                    reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_IN_PROGRESS,
                )

            acquired_at = self._clock()
            lease_deadline = acquired_at + timedelta(seconds=lease)
            reservation_id = self._reservation_id_factory()
            if not reservation_id:
                raise ValueError("reservation_id_factory must return non-empty value")

            new_reservation = ArtifactCreationReservation(
                reservation_id=reservation_id,
                artifact_lookup_key_hash=key_hash,
                tenant_id=lookup_key.tenant_id,
                owner_operation_id=owner,
                acquired_at=acquired_at,
                lease_deadline=lease_deadline,
            )
            self._reservations[state_key] = new_reservation
            self._increment_state_version_locked(state_key)
            self._condition.notify_all()
            return ArtifactCreationCoordinationResult(
                status=ArtifactCreationCoordinationStatus.ACQUIRED,
                artifact_lookup_key_hash=key_hash,
                state_version=self._state_version_locked(state_key),
                reservation=new_reservation,
            )

    def store_validated_artifact(
        self,
        *,
        reservation: ArtifactCreationReservation,
        artifact: StoredOptimizationArtifact,
    ) -> OptimizationArtifactReference:
        active_reservation = _require_reservation(reservation)
        stored_artifact = _require_stored_artifact(artifact)
        metadata = stored_artifact.metadata
        lookup_key = metadata.lookup_key
        key_hash = compute_artifact_lookup_key_hash(lookup_key)
        state_key = (lookup_key.tenant_id, key_hash)

        with self._lock:
            self._ensure_open()

            if metadata.status is not ReusableArtifactStatus.VALIDATED:
                raise ValueError("artifact metadata.status must be VALIDATED")
            if metadata.validation.status is not ArtifactValidationStatus.PASSED:
                raise ValueError("artifact validation.status must be PASSED")

            if active_reservation.tenant_id != lookup_key.tenant_id:
                raise ValueError("reservation tenant_id must match artifact lookup_key tenant_id")
            if active_reservation.artifact_lookup_key_hash != key_hash:
                raise ValueError("reservation artifact_lookup_key_hash must match lookup key hash")

            current_reservation = self._reservations.get(state_key)
            if current_reservation is None:
                raise RuntimeError(
                    ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
                )
            if current_reservation.reservation_id != active_reservation.reservation_id:
                raise ValueError("reservation_id does not match active reservation")
            if current_reservation.owner_operation_id != active_reservation.owner_operation_id:
                raise ValueError("owner_operation_id does not match active reservation")

            now = self._clock()
            if current_reservation.lease_deadline <= now:
                self._remove_reservation_locked(state_key)
                self._increment_state_version_locked(state_key)
                self._condition.notify_all()
                raise RuntimeError(ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED.value)

            existing_active = self._active_by_key.get(state_key)
            if existing_active is not None:
                existing_metadata = existing_active.metadata
                if (
                    existing_metadata.artifact_id == metadata.artifact_id
                    and existing_metadata.artifact_content_hash == metadata.artifact_content_hash
                    and existing_active.payload == stored_artifact.payload
                ):
                    self._remove_reservation_locked(state_key)
                    self._increment_state_version_locked(state_key)
                    self._condition.notify_all()
                    return build_optimization_artifact_reference(existing_active)
                raise RuntimeError(
                    ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
                )

            artifact_id_key = (lookup_key.tenant_id, metadata.artifact_id)
            if artifact_id_key in self._artifacts_by_id:
                raise RuntimeError(
                    ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
                )

            self._active_by_key[state_key] = stored_artifact
            self._artifacts_by_id[artifact_id_key] = stored_artifact
            self._canonical_keys[state_key] = lookup_key
            self._remove_reservation_locked(state_key)
            self._increment_state_version_locked(state_key)
            self._condition.notify_all()
            return build_optimization_artifact_reference(stored_artifact)

    def release_creation_reservation(
        self,
        *,
        reservation: ArtifactCreationReservation,
        reason_code: ContextOptimizationReasonCode | None = None,
    ) -> bool:
        active_reservation = _require_reservation(reservation)
        if reason_code is not None and not isinstance(reason_code, ContextOptimizationReasonCode):
            raise ValueError("reason_code must be ContextOptimizationReasonCode when provided")

        state_key = (
            active_reservation.tenant_id,
            active_reservation.artifact_lookup_key_hash,
        )

        with self._lock:
            self._ensure_open()
            current = self._reservations.get(state_key)
            if current is None:
                return False
            if (
                current.reservation_id != active_reservation.reservation_id
                or current.owner_operation_id != active_reservation.owner_operation_id
            ):
                return False

            self._remove_reservation_locked(state_key)
            self._increment_state_version_locked(state_key)
            self._condition.notify_all()
            return True

    def wait_for_artifact_or_reservation_change(
        self,
        key: ArtifactLookupKey,
        *,
        observed_state_version: int,
        timeout_seconds: float,
    ) -> bool:
        lookup_key = _require_lookup_key(key)
        observed = _require_non_negative_int(observed_state_version, "observed_state_version")
        timeout = _require_timeout_seconds(timeout_seconds)
        key_hash = compute_artifact_lookup_key_hash(lookup_key)
        state_key = (lookup_key.tenant_id, key_hash)

        with self._lock:
            self._ensure_open()
            return self._wait_for_change_locked(
                state_key=state_key,
                observed_state_version=observed,
                timeout_seconds=timeout,
            )

    def invalidate_artifact(
        self,
        reference: OptimizationArtifactReference,
        *,
        reason: str,
    ) -> StoredOptimizationArtifact | None:
        ref = _require_reference(reference)
        invalidation_reason = _require_non_empty(reason, "reason")
        return self._transition_artifact_status(
            ref,
            target_status=ReusableArtifactStatus.INVALIDATED,
            invalidation_reason=invalidation_reason,
        )

    def retire_artifact(
        self,
        reference: OptimizationArtifactReference,
        *,
        reason: str,
    ) -> StoredOptimizationArtifact | None:
        ref = _require_reference(reference)
        retirement_reason = _require_non_empty(reason, "reason")
        return self._transition_artifact_status(
            ref,
            target_status=ReusableArtifactStatus.RETIRED,
            invalidation_reason=retirement_reason,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._active_by_key.clear()
            self._artifacts_by_id.clear()
            self._canonical_keys.clear()
            self._reservations.clear()
            self._state_versions.clear()
            self._condition.notify_all()

    def _transition_artifact_status(
        self,
        reference: OptimizationArtifactReference,
        *,
        target_status: ReusableArtifactStatus,
        invalidation_reason: str,
    ) -> StoredOptimizationArtifact | None:
        with self._lock:
            self._ensure_open()
            stored = self._resolve_locked(reference)
            if stored is None:
                return None

            metadata = stored.metadata
            updated_metadata = ReusableOptimizationArtifact(
                artifact_id=metadata.artifact_id,
                lookup_key=metadata.lookup_key,
                artifact_content_hash=metadata.artifact_content_hash,
                created_at=metadata.created_at,
                created_by_executor=metadata.created_by_executor,
                validation=metadata.validation,
                status=target_status,
                invalidation_reason=invalidation_reason,
                supersedes_artifact_id=metadata.supersedes_artifact_id,
                receipt_ref=metadata.receipt_ref,
                safe_metadata=metadata.safe_metadata,
            )
            updated = StoredOptimizationArtifact(
                metadata=updated_metadata,
                payload=stored.payload,
                media_type=stored.media_type,
                encoding=stored.encoding,
            )

            tenant_id = metadata.lookup_key.tenant_id
            key_hash = compute_artifact_lookup_key_hash(metadata.lookup_key)
            state_key = (tenant_id, key_hash)
            artifact_id_key = (tenant_id, metadata.artifact_id)

            self._artifacts_by_id[artifact_id_key] = updated
            if self._active_by_key.get(state_key) is stored:
                del self._active_by_key[state_key]

            self._increment_state_version_locked(state_key)
            self._condition.notify_all()
            return updated

    def _lookup_eligible_locked(self, lookup_key: ArtifactLookupKey) -> StoredOptimizationArtifact | None:
        key_hash = compute_artifact_lookup_key_hash(lookup_key)
        state_key = (lookup_key.tenant_id, key_hash)
        stored = self._active_by_key.get(state_key)
        if stored is None:
            return None

        canonical_key = self._canonical_keys.get(state_key)
        if canonical_key is None:
            raise RuntimeError(
                ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
            )
        if canonical_key != lookup_key:
            raise RuntimeError(
                ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
            )

        if stored.metadata.status is not ReusableArtifactStatus.VALIDATED:
            return None
        if stored.metadata.validation.status is not ArtifactValidationStatus.PASSED:
            return None
        return stored

    def _resolve_locked(self, reference: OptimizationArtifactReference) -> StoredOptimizationArtifact | None:
        stored = self._artifacts_by_id.get((reference.tenant_id, reference.artifact_id))
        if stored is None:
            return None

        metadata = stored.metadata
        lookup_hash = compute_artifact_lookup_key_hash(metadata.lookup_key)
        if lookup_hash != reference.artifact_lookup_key_hash:
            return None
        if metadata.artifact_content_hash != reference.artifact_content_hash:
            return None
        if metadata.lookup_key.artifact_type != reference.artifact_type:
            return None
        return stored

    def _wait_for_change_locked(
        self,
        *,
        state_key: TenantKeyHash,
        observed_state_version: int,
        timeout_seconds: float,
    ) -> bool:
        remaining = timeout_seconds

        while True:
            current_version = self._state_version_locked(state_key)
            if current_version > observed_state_version:
                return True

            reservation = self._reservations.get(state_key)
            wait_timeout = remaining
            if reservation is not None:
                now = self._clock()
                lease_remaining = (reservation.lease_deadline - now).total_seconds()
                if lease_remaining <= 0:
                    self._remove_reservation_locked(state_key)
                    self._increment_state_version_locked(state_key)
                    self._condition.notify_all()
                    return True
                wait_timeout = min(wait_timeout, lease_remaining)

            if wait_timeout <= 0:
                reservation = self._reservations.get(state_key)
                if reservation is not None:
                    now = self._clock()
                    if reservation.lease_deadline <= now:
                        self._remove_reservation_locked(state_key)
                        self._increment_state_version_locked(state_key)
                        self._condition.notify_all()
                        return True
                if self._state_version_locked(state_key) > observed_state_version:
                    return True
                return False

            self._condition.wait(timeout=wait_timeout)
            self._ensure_open()
            remaining -= wait_timeout
            if remaining <= 0:
                reservation = self._reservations.get(state_key)
                if reservation is not None:
                    now = self._clock()
                    if reservation.lease_deadline <= now:
                        self._remove_reservation_locked(state_key)
                        self._increment_state_version_locked(state_key)
                        self._condition.notify_all()
                        return True
                if self._state_version_locked(state_key) > observed_state_version:
                    return True
                return False

    def _state_version_locked(self, state_key: TenantKeyHash) -> int:
        return self._state_versions.get(state_key, 0)

    def _increment_state_version_locked(self, state_key: TenantKeyHash) -> int:
        version = self._state_versions.get(state_key, 0) + 1
        self._state_versions[state_key] = version
        return version

    def _remove_reservation_locked(self, state_key: TenantKeyHash) -> None:
        self._reservations.pop(state_key, None)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(_CLOSED_ERROR)

    def _validate_clock(self) -> None:
        sample = self._clock()
        if sample.tzinfo is None or sample.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetime")

    def _validate_reservation_id_factory(self) -> None:
        reservation_id = self._reservation_id_factory()
        if not reservation_id:
            raise ValueError("reservation_id_factory must return non-empty value")
