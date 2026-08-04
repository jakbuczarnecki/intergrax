# © Artur Czarnecki. All rights reserved.

"""TOKEN-10E-4 proof for the durable UCL repository adapter."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.runtime.context_lifecycle import (
    ArtifactCreationCoordinationStatus,
    ArtifactLookupKey,
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactRepository,
    SQLiteOptimizationArtifactRepository,
)
from tests.unit.runtime.context_lifecycle.test_repository_contracts import (
    _lookup_key,
    _stored_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class FakeClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


@pytest.fixture(params=("memory", "sqlite"))
def repository(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        value = InMemoryOptimizationArtifactRepository()
    else:
        value = SQLiteOptimizationArtifactRepository(str(tmp_path / "artifacts.sqlite"))
    yield value
    value.close()


def _publish(repository: object, key: ArtifactLookupKey, *, artifact_id: str = "artifact-1") -> object:
    result = repository.try_acquire_creation_reservation(  # type: ignore[attr-defined]
        key,
        owner_operation_id="operation-1",
        lease_seconds=60,
    )
    assert result.reservation is not None
    return repository.store_validated_artifact(  # type: ignore[attr-defined]
        reservation=result.reservation,
        artifact=_stored_artifact(metadata={"artifact_id": artifact_id}),
    )


def test_contract_surface_and_tenant_scoped_resolution(repository: object) -> None:
    assert isinstance(repository, OptimizationArtifactRepository)
    key = _lookup_key()
    reference = _publish(repository, key)
    assert repository.lookup(key) is not None  # type: ignore[attr-defined]
    assert repository.resolve(reference) is not None  # type: ignore[attr-defined]
    assert repository.resolve(  # type: ignore[attr-defined]
        reference.__class__(
            tenant_id="tenant-2",
            artifact_id=reference.artifact_id,
            artifact_lookup_key_hash=reference.artifact_lookup_key_hash,
            artifact_content_hash=reference.artifact_content_hash,
            artifact_type=reference.artifact_type,
        )
    ) is None


def test_sqlite_restart_reopens_artifact_and_reservation(tmp_path: Path) -> None:
    db_path = str(tmp_path / "restart.sqlite")
    first = SQLiteOptimizationArtifactRepository(db_path)
    key = _lookup_key()
    reservation_result = first.try_acquire_creation_reservation(
        key,
        owner_operation_id="operation-1",
        lease_seconds=60,
    )
    assert reservation_result.reservation is not None
    stored = _stored_artifact()
    reference = first.store_validated_artifact(
        reservation=reservation_result.reservation,
        artifact=stored,
    )
    reservation_key = _lookup_key(source_refs=("reservation-after-restart",))
    persistent_reservation = first.try_acquire_creation_reservation(
        reservation_key,
        owner_operation_id="operation-reservation",
        lease_seconds=60,
    )
    assert persistent_reservation.reservation is not None
    first.close()

    second = SQLiteOptimizationArtifactRepository(db_path)
    assert second.lookup(key) == stored
    assert second.resolve(reference) == stored
    available = second.try_acquire_creation_reservation(
        key,
        owner_operation_id="operation-2",
        lease_seconds=60,
    )
    assert available.status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE
    in_progress = second.try_acquire_creation_reservation(
        reservation_key,
        owner_operation_id="operation-other",
        lease_seconds=60,
    )
    assert in_progress.status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS
    assert second.release_creation_reservation(reservation=persistent_reservation.reservation)
    second.close()


def test_sqlite_reservation_is_atomic_across_repository_instances(tmp_path: Path) -> None:
    db_path = str(tmp_path / "race.sqlite")
    first = SQLiteOptimizationArtifactRepository(db_path)
    second = SQLiteOptimizationArtifactRepository(db_path)
    key = _lookup_key()
    barrier = threading.Barrier(2)
    results: list[ArtifactCreationCoordinationStatus] = []

    def acquire(repository: SQLiteOptimizationArtifactRepository, owner: str) -> None:
        barrier.wait()
        result = repository.try_acquire_creation_reservation(
            key,
            owner_operation_id=owner,
            lease_seconds=60,
        )
        results.append(result.status)

    left = threading.Thread(target=acquire, args=(first, "operation-1"))
    right = threading.Thread(target=acquire, args=(second, "operation-2"))
    left.start()
    right.start()
    left.join()
    right.join()
    assert results.count(ArtifactCreationCoordinationStatus.ACQUIRED) == 1
    assert results.count(ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS) == 1
    first.close()
    second.close()


def test_sqlite_lease_expiry_and_owner_scoped_release(tmp_path: Path) -> None:
    clock = FakeClock()
    db_path = str(tmp_path / "lease.sqlite")
    repository = SQLiteOptimizationArtifactRepository(db_path, clock=clock.now)
    key = _lookup_key()
    acquired = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="operation-1",
        lease_seconds=10,
    )
    assert acquired.reservation is not None
    wrong_owner = acquired.reservation.__class__(
        reservation_id=acquired.reservation.reservation_id,
        tenant_id=acquired.reservation.tenant_id,
        artifact_lookup_key_hash=acquired.reservation.artifact_lookup_key_hash,
        owner_operation_id="operation-2",
        acquired_at=acquired.reservation.acquired_at,
        lease_deadline=acquired.reservation.lease_deadline,
    )
    assert not repository.release_creation_reservation(reservation=wrong_owner)
    clock.advance(11)
    expired = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="operation-2",
        lease_seconds=60,
    )
    assert expired.status is ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED
    reacquired = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="operation-2",
        lease_seconds=60,
    )
    assert reacquired.status is ArtifactCreationCoordinationStatus.ACQUIRED
    assert repository.release_creation_reservation(reservation=reacquired.reservation)
    repository.close()


def test_sqlite_store_requires_matching_reservation_and_preserves_lifecycle(tmp_path: Path) -> None:
    repository = SQLiteOptimizationArtifactRepository(str(tmp_path / "lifecycle.sqlite"))
    key = _lookup_key()
    acquired = repository.try_acquire_creation_reservation(
        key,
        owner_operation_id="operation-1",
        lease_seconds=60,
    )
    assert acquired.reservation is not None
    wrong = _stored_artifact(metadata={"artifact_id": "different"})
    with pytest.raises(ValueError, match="reservation"):
        repository.store_validated_artifact(
            reservation=acquired.reservation.__class__(
                reservation_id=acquired.reservation.reservation_id,
                tenant_id="tenant-2",
                artifact_lookup_key_hash=acquired.reservation.artifact_lookup_key_hash,
                owner_operation_id=acquired.reservation.owner_operation_id,
                acquired_at=acquired.reservation.acquired_at,
                lease_deadline=acquired.reservation.lease_deadline,
            ),
            artifact=wrong,
        )
    reference = repository.store_validated_artifact(
        reservation=acquired.reservation,
        artifact=_stored_artifact(),
    )
    assert repository.store_validated_artifact(
        reservation=acquired.reservation,
        artifact=_stored_artifact(),
    ) == reference
    invalidated = repository.invalidate_artifact(reference, reason="test")
    assert invalidated is not None
    assert repository.lookup(key) is None
    assert repository.resolve(reference) is not None
    assert repository.retire_artifact(reference, reason="retire") is not None
    assert not repository.wait_for_artifact_or_reservation_change(
        key,
        observed_state_version=10,
        timeout_seconds=0,
    )
    repository.close()
