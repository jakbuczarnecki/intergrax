# © Artur Czarnecki. All rights reserved.

"""Optimization artifact repository contracts (CTX-UCL-2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCreationCoordinationStatus,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ContextOptimizationReasonCode,
    OptimizationArtifactType,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_non_negative(value: object, field_name: str) -> int:
    int_value = _require_int(value, field_name)
    if int_value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return int_value


def _require_instance(value: object, expected_type: type, field_name: str) -> object:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name} must be {expected_type.__name__}")
    return value


def compute_artifact_content_hash(payload: bytes) -> str:
    """Return SHA-256 lowercase hex digest for opaque artifact payload bytes."""
    if type(payload) is not bytes:
        raise ValueError("payload must be bytes")
    if not payload:
        raise ValueError("payload must be non-empty")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class OptimizationArtifactRepositoryCapabilities:
    """Declared backend capabilities for an optimization artifact repository."""

    backend_id: str
    durable: bool
    shared_across_processes: bool
    supports_single_flight: bool
    supports_bounded_wait: bool
    reference_only: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_id", _require_non_empty(self.backend_id, "backend_id"))
        object.__setattr__(self, "durable", _require_bool(self.durable, "durable"))
        object.__setattr__(
            self,
            "shared_across_processes",
            _require_bool(self.shared_across_processes, "shared_across_processes"),
        )
        object.__setattr__(
            self,
            "supports_single_flight",
            _require_bool(self.supports_single_flight, "supports_single_flight"),
        )
        object.__setattr__(
            self,
            "supports_bounded_wait",
            _require_bool(self.supports_bounded_wait, "supports_bounded_wait"),
        )
        object.__setattr__(self, "reference_only", _require_bool(self.reference_only, "reference_only"))


@dataclass(frozen=True, slots=True)
class StoredOptimizationArtifact:
    """Repository envelope separating metadata from opaque payload bytes."""

    metadata: ReusableOptimizationArtifact
    payload: bytes = field(repr=False)
    media_type: str
    encoding: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _require_instance(self.metadata, ReusableOptimizationArtifact, "metadata"),
        )
        if type(self.payload) is not bytes:
            raise ValueError("payload must be bytes")
        if not self.payload:
            raise ValueError("payload must be non-empty")
        object.__setattr__(self, "media_type", _require_non_empty(self.media_type, "media_type"))
        if self.encoding is not None:
            object.__setattr__(
                self,
                "encoding",
                _require_non_empty(self.encoding, "encoding"),
            )

        metadata = self.metadata
        if metadata.status is ReusableArtifactStatus.VALIDATED:
            if metadata.validation.status is not ArtifactValidationStatus.PASSED:
                raise ValueError("metadata.validation.status must be PASSED")

        content_hash = compute_artifact_content_hash(self.payload)
        if content_hash != metadata.artifact_content_hash:
            raise ValueError("payload SHA-256 must equal metadata.artifact_content_hash")


@dataclass(frozen=True, slots=True)
class OptimizationArtifactReference:
    """Immutable artifact reference for revision linkage and audit."""

    tenant_id: str
    artifact_id: str
    artifact_lookup_key_hash: str
    artifact_content_hash: str
    artifact_type: OptimizationArtifactType

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _require_non_empty(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "artifact_id", _require_non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "artifact_lookup_key_hash",
            _require_non_empty(self.artifact_lookup_key_hash, "artifact_lookup_key_hash"),
        )
        object.__setattr__(
            self,
            "artifact_content_hash",
            _require_non_empty(self.artifact_content_hash, "artifact_content_hash"),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _require_instance(self.artifact_type, OptimizationArtifactType, "artifact_type"),
        )


def build_optimization_artifact_reference(
    artifact: StoredOptimizationArtifact,
) -> OptimizationArtifactReference:
    """Build a tenant-scoped artifact reference from stored artifact envelope."""
    from intergrax.runtime.context_lifecycle.serialization import compute_artifact_lookup_key_hash

    metadata = artifact.metadata
    lookup_hash = compute_artifact_lookup_key_hash(metadata.lookup_key)
    return OptimizationArtifactReference(
        tenant_id=metadata.lookup_key.tenant_id,
        artifact_id=metadata.artifact_id,
        artifact_lookup_key_hash=lookup_hash,
        artifact_content_hash=metadata.artifact_content_hash,
        artifact_type=metadata.lookup_key.artifact_type,
    )


@dataclass(frozen=True, slots=True)
class ArtifactCreationCoordinationResult:
    """Repository coordination outcome for single-flight artifact creation."""

    status: ArtifactCreationCoordinationStatus
    artifact_lookup_key_hash: str
    state_version: int
    reservation: ArtifactCreationReservation | None = None
    artifact_reference: OptimizationArtifactReference | None = None
    reason_code: ContextOptimizationReasonCode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _require_instance(
                self.status,
                ArtifactCreationCoordinationStatus,
                "status",
            ),
        )
        object.__setattr__(
            self,
            "artifact_lookup_key_hash",
            _require_non_empty(self.artifact_lookup_key_hash, "artifact_lookup_key_hash"),
        )
        object.__setattr__(
            self,
            "state_version",
            _require_non_negative(self.state_version, "state_version"),
        )

        if self.reservation is not None:
            object.__setattr__(
                self,
                "reservation",
                _require_instance(
                    self.reservation,
                    ArtifactCreationReservation,
                    "reservation",
                ),
            )
        if self.artifact_reference is not None:
            object.__setattr__(
                self,
                "artifact_reference",
                _require_instance(
                    self.artifact_reference,
                    OptimizationArtifactReference,
                    "artifact_reference",
                ),
            )
        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                _require_instance(
                    self.reason_code,
                    ContextOptimizationReasonCode,
                    "reason_code",
                ),
            )

        status = self.status
        if status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE:
            if self.artifact_reference is None:
                raise ValueError("ARTIFACT_AVAILABLE requires artifact_reference")
            if self.reservation is not None:
                raise ValueError("ARTIFACT_AVAILABLE requires reservation is None")
            if self.reason_code is not None:
                raise ValueError("ARTIFACT_AVAILABLE requires reason_code is None")
        elif status is ArtifactCreationCoordinationStatus.ACQUIRED:
            if self.reservation is None:
                raise ValueError("ACQUIRED requires reservation")
            if self.artifact_reference is not None:
                raise ValueError("ACQUIRED requires artifact_reference is None")
            if self.reason_code is not None:
                raise ValueError("ACQUIRED requires reason_code is None")
        elif status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS:
            if self.reservation is None:
                raise ValueError("ALREADY_IN_PROGRESS requires reservation")
            if self.artifact_reference is not None:
                raise ValueError("ALREADY_IN_PROGRESS requires artifact_reference is None")
            if self.reason_code is not ContextOptimizationReasonCode.ARTIFACT_CREATION_IN_PROGRESS:
                raise ValueError(
                    "ALREADY_IN_PROGRESS requires reason_code == ARTIFACT_CREATION_IN_PROGRESS"
                )
        elif status is ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED:
            if self.reservation is None:
                raise ValueError("RESERVATION_EXPIRED requires reservation")
            if self.artifact_reference is not None:
                raise ValueError("RESERVATION_EXPIRED requires artifact_reference is None")
            if self.reason_code is not ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED:
                raise ValueError(
                    "RESERVATION_EXPIRED requires reason_code == ARTIFACT_CREATION_LEASE_EXPIRED"
                )
        elif status is ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT:
            if self.artifact_reference is not None:
                raise ValueError("RESERVATION_CONFLICT requires artifact_reference is None")
            if self.reason_code is not ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT:
                raise ValueError(
                    "RESERVATION_CONFLICT requires "
                    "reason_code == ARTIFACT_CREATION_RESERVATION_CONFLICT"
                )


@runtime_checkable
class OptimizationArtifactRepository(Protocol):
    """Backend-neutral optimization artifact repository port."""

    @property
    def capabilities(self) -> OptimizationArtifactRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def lookup(self, key: ArtifactLookupKey) -> StoredOptimizationArtifact | None:
        """Return an eligible validated artifact for an exact lookup key or None."""

    def resolve(self, reference: OptimizationArtifactReference) -> StoredOptimizationArtifact | None:
        """Resolve an artifact reference regardless of lifecycle status."""

    def try_acquire_creation_reservation(
        self,
        key: ArtifactLookupKey,
        *,
        owner_operation_id: str,
        lease_seconds: int,
    ) -> ArtifactCreationCoordinationResult:
        """Atomically coordinate single-flight artifact creation reservation."""

    def store_validated_artifact(
        self,
        *,
        reservation: ArtifactCreationReservation,
        artifact: StoredOptimizationArtifact,
    ) -> OptimizationArtifactReference:
        """Publish a validated artifact and complete the active reservation."""

    def release_creation_reservation(
        self,
        *,
        reservation: ArtifactCreationReservation,
        reason_code: ContextOptimizationReasonCode | None = None,
    ) -> bool:
        """Release an active creation reservation without storing an artifact."""

    def wait_for_artifact_or_reservation_change(
        self,
        key: ArtifactLookupKey,
        *,
        observed_state_version: int,
        timeout_seconds: float,
    ) -> bool:
        """Bounded wait for observable state change on a lookup key."""

    def invalidate_artifact(
        self,
        reference: OptimizationArtifactReference,
        *,
        reason: str,
    ) -> StoredOptimizationArtifact | None:
        """Invalidate an artifact while preserving payload and history."""

    def retire_artifact(
        self,
        reference: OptimizationArtifactReference,
        *,
        reason: str,
    ) -> StoredOptimizationArtifact | None:
        """Retire an artifact while preserving payload and history."""

    def close(self) -> None:
        """Release process-local repository resources."""
