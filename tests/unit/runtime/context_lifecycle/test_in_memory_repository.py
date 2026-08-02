# © Artur Czarnecki. All rights reserved.

"""In-memory optimization artifact repository tests (CTX-UCL-2)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.context_lifecycle import (
    ArtifactCreationCoordinationStatus,
    ArtifactCreationReservation,
    ArtifactCompressionTarget,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    ContextOptimizationReasonCode,
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactReference,
    OptimizationArtifactType,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
    compute_artifact_content_hash,
    compute_artifact_lookup_key_hash,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BASE_TIME = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)


def _lookup_key(**overrides: object) -> ArtifactLookupKey:
    defaults: dict[str, object] = {
        "tenant_id": "tenant-1",
        "context_scope_id": "scope-1",
        "artifact_type": OptimizationArtifactType.MESSAGE_SEQUENCE,
        "source_content_hash": "hash-abc",
        "strategy_id": "strategy.summarize",
        "strategy_version": "1.0.0",
        "policy_version": "policy-v1",
        "validation_contract_version": "validation-v1",
        "compression_target": ArtifactCompressionTarget(target_tokens=1000),
        "lossiness_profile": "lossy_summary",
        "source_refs": ("msg-1", "msg-2"),
    }
    defaults.update(overrides)
    return ArtifactLookupKey(**defaults)  # type: ignore[arg-type]


def _validation_summary(**overrides: object) -> ArtifactValidationSummary:
    defaults: dict[str, object] = {
        "status": ArtifactValidationStatus.PASSED,
        "validation_contract_version": "validation-v1",
        "validated_at": _BASE_TIME,
    }
    defaults.update(overrides)
    return ArtifactValidationSummary(**defaults)  # type: ignore[arg-type]


def _reusable_artifact(**overrides: object) -> ReusableOptimizationArtifact:
    payload = overrides.pop("payload", b"payload-bytes")
    defaults: dict[str, object] = {
        "artifact_id": "artifact-1",
        "lookup_key": _lookup_key(),
        "artifact_content_hash": compute_artifact_content_hash(payload),
        "created_at": _BASE_TIME,
        "created_by_executor": "executor.message_sequence",
        "validation": _validation_summary(),
    }
    defaults.update(overrides)
    return ReusableOptimizationArtifact(**defaults)  # type: ignore[arg-type]


def _stored_artifact(
    payload: bytes = b"payload-bytes",
    lookup_key: ArtifactLookupKey | None = None,
    **metadata_overrides: object,
) -> StoredOptimizationArtifact:
    metadata = _reusable_artifact(
        payload=payload,
        lookup_key=lookup_key or _lookup_key(),
        **metadata_overrides,
    )
    return StoredOptimizationArtifact(
        metadata=metadata,
        payload=payload,
        media_type="application/octet-stream",
    )


class FakeClock:
    def __init__(self, start: datetime = _BASE_TIME) -> None:
        self._current = start
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._current

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._current = self._current + timedelta(seconds=seconds)


def _repository(
    clock: FakeClock | None = None,
    reservation_ids: list[str] | None = None,
) -> InMemoryOptimizationArtifactRepository:
    fake_clock = clock or FakeClock()
    ids = reservation_ids or []
    counter = 0

    def reservation_id_factory() -> str:
        nonlocal counter
        if counter < len(ids):
            value = ids[counter]
        else:
            value = f"reservation-{counter}"
        counter += 1
        return value

    return InMemoryOptimizationArtifactRepository(
        clock=fake_clock.now,
        reservation_id_factory=reservation_id_factory,
    )


def _acquire(
    repository: InMemoryOptimizationArtifactRepository,
    key: ArtifactLookupKey,
    owner: str = "owner-1",
    lease_seconds: int = 60,
) -> ArtifactCreationReservation:
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id=owner,
        lease_seconds=lease_seconds,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ACQUIRED
    assert result.reservation is not None
    return result.reservation


def _publish(
    repository: InMemoryOptimizationArtifactRepository,
    reservation: ArtifactCreationReservation,
    key: ArtifactLookupKey | None = None,
    payload: bytes = b"payload-bytes",
    artifact_id: str = "artifact-1",
) -> OptimizationArtifactReference:
    lookup_key = key or _lookup_key()
    stored = _stored_artifact(payload=payload, lookup_key=lookup_key, artifact_id=artifact_id)
    return repository.store_validated_artifact(reservation=reservation, artifact=stored)


def test_lookup_exact_hit() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    found = repository.lookup(key)
    assert found is not None
    assert found.payload == b"payload-bytes"
    repository.close()


def test_lookup_missing_returns_none() -> None:
    repository = _repository()
    assert repository.lookup(_lookup_key()) is None
    repository.close()


def test_lookup_source_hash_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(_lookup_key(source_content_hash="other-hash")) is None
    repository.close()


def test_lookup_strategy_id_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(_lookup_key(strategy_id="other.strategy")) is None
    repository.close()


def test_lookup_strategy_version_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(_lookup_key(strategy_version="9.9.9")) is None
    repository.close()


def test_lookup_policy_version_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(_lookup_key(policy_version="policy-v2")) is None
    repository.close()


def test_lookup_validation_contract_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(_lookup_key(validation_contract_version="validation-v2")) is None
    repository.close()


def test_lookup_compression_target_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(
        _lookup_key(compression_target=ArtifactCompressionTarget(target_tokens=500))
    ) is None
    repository.close()


def test_lookup_tenant_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(_lookup_key(tenant_id="tenant-2")) is None
    repository.close()


def test_lookup_context_scope_miss() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    assert repository.lookup(_lookup_key(context_scope_id="scope-2")) is None
    repository.close()


def test_lookup_invalidated_not_reusable() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.invalidate_artifact(reference, reason="stale")
    assert repository.lookup(key) is None
    repository.close()


def test_lookup_retired_not_reusable() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.retire_artifact(reference, reason="retired")
    assert repository.lookup(key) is None
    repository.close()


def test_resolve_validated_artifact() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    resolved = repository.resolve(reference)
    assert resolved is not None
    assert resolved.payload == b"payload-bytes"
    repository.close()


def test_resolve_invalidated_artifact() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.invalidate_artifact(reference, reason="stale")
    resolved = repository.resolve(reference)
    assert resolved is not None
    assert resolved.metadata.status is ReusableArtifactStatus.INVALIDATED
    repository.close()


def test_resolve_retired_artifact() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.retire_artifact(reference, reason="retired")
    resolved = repository.resolve(reference)
    assert resolved is not None
    assert resolved.metadata.status is ReusableArtifactStatus.RETIRED
    repository.close()


def test_resolve_cross_tenant_returns_none() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    cross_tenant = OptimizationArtifactReference(
        tenant_id="tenant-2",
        artifact_id=reference.artifact_id,
        artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
        artifact_content_hash=reference.artifact_content_hash,
        artifact_type=reference.artifact_type,
    )
    assert repository.resolve(cross_tenant) is None
    repository.close()


def test_resolve_tampered_lookup_hash() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    tampered = OptimizationArtifactReference(
        tenant_id=reference.tenant_id,
        artifact_id=reference.artifact_id,
        artifact_lookup_key_hash="tampered-hash",
        artifact_content_hash=reference.artifact_content_hash,
        artifact_type=reference.artifact_type,
    )
    assert repository.resolve(tampered) is None
    repository.close()


def test_resolve_tampered_content_hash() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    tampered = OptimizationArtifactReference(
        tenant_id=reference.tenant_id,
        artifact_id=reference.artifact_id,
        artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
        artifact_content_hash="tampered-content-hash",
        artifact_type=reference.artifact_type,
    )
    assert repository.resolve(tampered) is None
    repository.close()


def test_resolve_tampered_artifact_type() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    tampered = OptimizationArtifactReference(
        tenant_id=reference.tenant_id,
        artifact_id=reference.artifact_id,
        artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
        artifact_content_hash=reference.artifact_content_hash,
        artifact_type=OptimizationArtifactType.TEXT,
    )
    assert repository.resolve(tampered) is None
    repository.close()


def test_reservation_first_caller_acquired() -> None:
    repository = _repository()
    key = _lookup_key()
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ACQUIRED
    repository.close()


def test_reservation_same_owner_replay() -> None:
    repository = _repository(reservation_ids=["res-1", "res-2"])
    key = _lookup_key()
    first = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    second = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    assert second.status is ArtifactCreationCoordinationStatus.ACQUIRED
    assert second.reservation == first.reservation
    repository.close()


def test_reservation_different_owner_in_progress() -> None:
    repository = _repository()
    key = _lookup_key()
    repository.try_acquire_creation_reservation(key, owner_operation_id="owner-1", lease_seconds=60)
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS
    assert result.reason_code is ContextOptimizationReasonCode.ARTIFACT_CREATION_IN_PROGRESS
    repository.close()


def test_reservation_existing_artifact_available() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE
    assert result.artifact_reference == reference
    repository.close()


def test_reservation_different_keys_independent() -> None:
    repository = _repository()
    key_a = _lookup_key(source_refs=("msg-a",))
    key_b = _lookup_key(source_refs=("msg-b",))
    result_a = repository.try_acquire_creation_reservation(
        key_a,
        owner_operation_id="owner-a",
        lease_seconds=60,
    )
    result_b = repository.try_acquire_creation_reservation(
        key_b,
        owner_operation_id="owner-b",
        lease_seconds=60,
    )
    assert result_a.status is ArtifactCreationCoordinationStatus.ACQUIRED
    assert result_b.status is ArtifactCreationCoordinationStatus.ACQUIRED
    repository.close()


def test_reservation_different_tenants_independent() -> None:
    repository = _repository()
    key_a = _lookup_key(tenant_id="tenant-a")
    key_b = _lookup_key(tenant_id="tenant-b")
    result_a = repository.try_acquire_creation_reservation(
        key_a,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    result_b = repository.try_acquire_creation_reservation(
        key_b,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    assert result_a.status is ArtifactCreationCoordinationStatus.ACQUIRED
    assert result_b.status is ArtifactCreationCoordinationStatus.ACQUIRED
    repository.close()


def test_reservation_lease_seconds_rejects_bool() -> None:
    repository = _repository()
    with pytest.raises(ValueError, match="lease_seconds"):
        repository.try_acquire_creation_reservation(
            _lookup_key(),
            owner_operation_id="owner-1",
            lease_seconds=True,  # type: ignore[arg-type]
        )
    repository.close()


def test_reservation_lease_seconds_rejects_zero() -> None:
    repository = _repository()
    with pytest.raises(ValueError, match="lease_seconds"):
        repository.try_acquire_creation_reservation(
            _lookup_key(),
            owner_operation_id="owner-1",
            lease_seconds=0,
        )
    repository.close()


def test_reservation_expired_produces_expired_status() -> None:
    clock = FakeClock()
    repository = _repository(clock=clock)
    key = _lookup_key()
    repository.try_acquire_creation_reservation(key, owner_operation_id="owner-1", lease_seconds=10)
    clock.advance(11)
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED
    assert result.reason_code is ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED
    repository.close()


def test_reservation_retry_after_expiry_acquires() -> None:
    clock = FakeClock()
    repository = _repository(clock=clock)
    key = _lookup_key()
    repository.try_acquire_creation_reservation(key, owner_operation_id="owner-1", lease_seconds=10)
    clock.advance(11)
    expired = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert expired.status is ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED
    acquired = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert acquired.status is ArtifactCreationCoordinationStatus.ACQUIRED
    repository.close()


def test_release_allows_later_acquisition() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key, owner="owner-1")
    assert repository.release_creation_reservation(reservation=reservation)
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ACQUIRED
    repository.close()


def test_release_wrong_reservation_returns_false() -> None:
    repository = _repository(reservation_ids=["res-1", "res-2"])
    key = _lookup_key()
    active = _acquire(repository, key, owner="owner-1")
    wrong = ArtifactCreationReservation(
        reservation_id="res-2",
        artifact_lookup_key_hash=compute_artifact_lookup_key_hash(key),
        tenant_id=key.tenant_id,
        owner_operation_id="owner-2",
        acquired_at=_BASE_TIME,
        lease_deadline=_BASE_TIME + timedelta(seconds=60),
    )
    assert not repository.release_creation_reservation(reservation=wrong)
    assert repository.release_creation_reservation(reservation=active)
    repository.close()


def test_store_matching_reservation() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    assert repository.lookup(key) is not None
    assert reference.artifact_id == "artifact-1"
    repository.close()


def test_store_removes_reservation() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key)
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE
    repository.close()


def test_store_wakes_waiter() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key, owner="owner-1")
    observed = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    ).state_version
    waiter_result: list[bool] = []

    def waiter() -> None:
        waiter_result.append(
            repository.wait_for_artifact_or_reservation_change(
                key,
                observed_state_version=observed,
                timeout_seconds=5.0,
            )
        )

    thread = threading.Thread(target=waiter)
    thread.start()
    _publish(repository, reservation, key)
    thread.join(timeout=5.0)
    assert waiter_result == [True]
    repository.close()


def test_store_tenant_mismatch_rejected() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    wrong_reservation = ArtifactCreationReservation(
        reservation_id=reservation.reservation_id,
        artifact_lookup_key_hash=reservation.artifact_lookup_key_hash,
        tenant_id="tenant-2",
        owner_operation_id=reservation.owner_operation_id,
        acquired_at=reservation.acquired_at,
        lease_deadline=reservation.lease_deadline,
    )
    with pytest.raises(ValueError, match="tenant_id"):
        repository.store_validated_artifact(
            reservation=wrong_reservation,
            artifact=_stored_artifact(lookup_key=key),
        )
    repository.close()


def test_store_lookup_key_mismatch_rejected() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    other_key = _lookup_key(source_refs=("msg-x",))
    with pytest.raises(ValueError, match="lookup key hash"):
        repository.store_validated_artifact(
            reservation=reservation,
            artifact=_stored_artifact(lookup_key=other_key),
        )
    repository.close()


def test_store_reservation_id_mismatch_rejected() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    wrong = ArtifactCreationReservation(
        reservation_id="wrong-id",
        artifact_lookup_key_hash=reservation.artifact_lookup_key_hash,
        tenant_id=reservation.tenant_id,
        owner_operation_id=reservation.owner_operation_id,
        acquired_at=reservation.acquired_at,
        lease_deadline=reservation.lease_deadline,
    )
    with pytest.raises(ValueError, match="reservation_id"):
        repository.store_validated_artifact(
            reservation=wrong,
            artifact=_stored_artifact(lookup_key=key),
        )
    repository.close()


def test_store_owner_mismatch_rejected() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key, owner="owner-1")
    wrong = ArtifactCreationReservation(
        reservation_id=reservation.reservation_id,
        artifact_lookup_key_hash=reservation.artifact_lookup_key_hash,
        tenant_id=reservation.tenant_id,
        owner_operation_id="owner-2",
        acquired_at=reservation.acquired_at,
        lease_deadline=reservation.lease_deadline,
    )
    with pytest.raises(ValueError, match="owner_operation_id"):
        repository.store_validated_artifact(
            reservation=wrong,
            artifact=_stored_artifact(lookup_key=key),
        )
    repository.close()


def test_store_expired_lease_rejected() -> None:
    clock = FakeClock()
    repository = _repository(clock=clock)
    key = _lookup_key()
    reservation = _acquire(repository, key, lease_seconds=10)
    clock.advance(11)
    with pytest.raises(RuntimeError, match="artifact_creation_lease_expired"):
        repository.store_validated_artifact(
            reservation=reservation,
            artifact=_stored_artifact(lookup_key=key),
        )
    repository.close()


def test_store_unvalidated_artifact_rejected() -> None:
    with pytest.raises(ValueError, match="PASSED"):
        _reusable_artifact(
            validation=_validation_summary(status=ArtifactValidationStatus.FAILED),
        )


def test_store_payload_hash_mismatch_rejected() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    metadata = _reusable_artifact(lookup_key=key, artifact_content_hash="wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        repository.store_validated_artifact(
            reservation=reservation,
            artifact=StoredOptimizationArtifact(
                metadata=metadata,
                payload=b"payload-bytes",
                media_type="application/octet-stream",
            ),
        )
    repository.close()


def test_store_conflicting_publication_rejected() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    _publish(repository, reservation, key, payload=b"first")
    with pytest.raises(RuntimeError, match="artifact_creation_reservation_conflict"):
        repository.store_validated_artifact(
            reservation=reservation,
            artifact=_stored_artifact(lookup_key=key, payload=b"second"),
        )
    repository.close()


def test_store_idempotent_replay() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    stored = _stored_artifact(lookup_key=key)
    first = repository.store_validated_artifact(reservation=reservation, artifact=stored)
    second = build_optimization_artifact_reference(stored)
    assert first == second
    repository.close()


def test_invalidation_removes_lookup_eligibility() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.invalidate_artifact(reference, reason="stale")
    assert repository.lookup(key) is None
    repository.close()


def test_invalidation_preserves_resolution() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.invalidate_artifact(reference, reason="stale")
    assert repository.resolve(reference) is not None
    repository.close()


def test_invalidation_preserves_payload() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key, payload=b"keep-payload")
    repository.invalidate_artifact(reference, reason="stale")
    resolved = repository.resolve(reference)
    assert resolved is not None
    assert resolved.payload == b"keep-payload"
    repository.close()


def test_retirement_removes_lookup_eligibility() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.retire_artifact(reference, reason="retired")
    assert repository.lookup(key) is None
    repository.close()


def test_retirement_preserves_resolution() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.retire_artifact(reference, reason="retired")
    assert repository.resolve(reference) is not None
    repository.close()


def test_invalidation_reason_required() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    with pytest.raises(ValueError, match="reason"):
        repository.invalidate_artifact(reference, reason="")
    repository.close()


def test_invalidation_cross_tenant_no_effect() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    cross_tenant = OptimizationArtifactReference(
        tenant_id="tenant-2",
        artifact_id=reference.artifact_id,
        artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
        artifact_content_hash=reference.artifact_content_hash,
        artifact_type=reference.artifact_type,
    )
    assert repository.invalidate_artifact(cross_tenant, reason="stale") is None
    assert repository.lookup(key) is not None
    repository.close()


def test_new_reservation_after_invalidation() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    repository.invalidate_artifact(reference, reason="stale")
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ACQUIRED
    repository.close()


def test_wait_unchanged_state_times_out() -> None:
    repository = _repository()
    key = _lookup_key()
    assert not repository.wait_for_artifact_or_reservation_change(
        key,
        observed_state_version=0,
        timeout_seconds=0.01,
    )
    repository.close()


def test_wait_already_changed_returns_immediately() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    observed = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-1",
        lease_seconds=60,
    ).state_version
    _publish(repository, reservation, key)
    assert repository.wait_for_artifact_or_reservation_change(
        key,
        observed_state_version=observed,
        timeout_seconds=1.0,
    )
    repository.close()


def test_wait_wakes_after_release() -> None:
    repository = _repository()
    key = _lookup_key()
    acquire_result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    reservation = acquire_result.reservation
    assert reservation is not None
    observed = acquire_result.state_version
    waiter_result: list[bool] = []

    def waiter() -> None:
        waiter_result.append(
            repository.wait_for_artifact_or_reservation_change(
                key,
                observed_state_version=observed,
                timeout_seconds=5.0,
            )
        )

    thread = threading.Thread(target=waiter)
    thread.start()
    repository.release_creation_reservation(reservation=reservation)
    thread.join(timeout=5.0)
    assert waiter_result == [True]
    repository.close()


def test_wait_wakes_after_invalidation() -> None:
    repository = _repository()
    key = _lookup_key()
    reservation = _acquire(repository, key)
    reference = _publish(repository, reservation, key)
    observed = 0
    waiter_result: list[bool] = []

    def waiter() -> None:
        waiter_result.append(
            repository.wait_for_artifact_or_reservation_change(
                key,
                observed_state_version=observed,
                timeout_seconds=5.0,
            )
        )

    thread = threading.Thread(target=waiter)
    thread.start()
    repository.invalidate_artifact(reference, reason="stale")
    thread.join(timeout=5.0)
    assert waiter_result == [True]
    repository.close()


def test_wait_observes_lease_expiry_without_background_thread() -> None:
    repository = InMemoryOptimizationArtifactRepository()
    key = _lookup_key()
    acquired = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-1",
        lease_seconds=1,
    )
    assert repository.wait_for_artifact_or_reservation_change(
        key,
        observed_state_version=acquired.state_version,
        timeout_seconds=5.0,
    )
    result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-2",
        lease_seconds=60,
    )
    assert result.status is ArtifactCreationCoordinationStatus.ACQUIRED
    repository.close()


def test_close_wakes_waiters() -> None:
    repository = _repository()
    key = _lookup_key()
    acquired_result = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="owner-1",
        lease_seconds=60,
    )
    observed_version = acquired_result.state_version
    waiter_errors: list[Exception] = []

    def waiter() -> None:
        try:
            repository.wait_for_artifact_or_reservation_change(
                key,
                observed_state_version=observed_version,
                timeout_seconds=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            waiter_errors.append(exc)

    thread = threading.Thread(target=waiter)
    thread.start()
    repository.close()
    thread.join(timeout=5.0)
    assert any(isinstance(exc, RuntimeError) for exc in waiter_errors)


def test_no_implicit_singleton_in_module() -> None:
    import intergrax.runtime.context_lifecycle.in_memory_repository as module

    assert not hasattr(module, "get_global_repository")
    assert not hasattr(module, "create_default_repository")


def test_no_nexus_import_of_in_memory_repository() -> None:
    import importlib
    import importlib.util

    nexus_spec = importlib.util.find_spec("intergrax.runtime.nexus")
    if nexus_spec is None:
        return
    import intergrax.runtime.nexus as nexus_module

    source = open(nexus_module.__file__, encoding="utf-8").read()
    assert "InMemoryOptimizationArtifactRepository" not in source


def test_no_token_optimization_import_of_in_memory_repository() -> None:
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("intergrax.runtime.token_optimization")
    if spec is None:
        return
    import intergrax.runtime.token_optimization as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "InMemoryOptimizationArtifactRepository" not in source
