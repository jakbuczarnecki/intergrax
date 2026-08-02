# © Artur Czarnecki. All rights reserved.

"""Concurrency tests for in-memory optimization artifact repository (CTX-UCL-2)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from intergrax.runtime.context_lifecycle import (
    ArtifactCreationCoordinationResult,
    ArtifactCreationCoordinationStatus,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    InMemoryOptimizationArtifactRepository,
    ReusableOptimizationArtifact,
    StoredOptimizationArtifact,
    compute_artifact_content_hash,
)
from tests.unit.runtime.context_lifecycle.test_in_memory_repository import (
    _lookup_key,
    _publish,
    _repository,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BASE_TIME = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)


def _stored_artifact(key: ArtifactLookupKey, payload: bytes = b"race-payload") -> StoredOptimizationArtifact:
    metadata = ReusableOptimizationArtifact(
        artifact_id="artifact-race",
        lookup_key=key,
        artifact_content_hash=compute_artifact_content_hash(payload),
        created_at=_BASE_TIME,
        created_by_executor="executor.message_sequence",
        validation=ArtifactValidationSummary(
            status=ArtifactValidationStatus.PASSED,
            validation_contract_version=key.validation_contract_version,
            validated_at=_BASE_TIME,
        ),
    )
    return StoredOptimizationArtifact(
        metadata=metadata,
        payload=payload,
        media_type="application/octet-stream",
    )


@pytest.mark.parametrize("iteration", range(5))
def test_same_key_race_exactly_one_acquired(iteration: int) -> None:
    repository = _repository()
    key = _lookup_key()
    barrier = threading.Barrier(2)
    results: list[ArtifactCreationCoordinationStatus] = []
    reservations: list[object] = []
    lock = threading.Lock()

    def contender(owner: str) -> None:
        barrier.wait()
        result = repository.try_acquire_creation_reservation(
            key,
            owner_operation_id=owner,
            lease_seconds=60,
        )
        with lock:
            results.append(result.status)
            if result.reservation is not None:
                reservations.append(result.reservation)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(contender, "owner-a")
        executor.submit(contender, "owner-b")
        executor.shutdown(wait=True)

    assert results.count(ArtifactCreationCoordinationStatus.ACQUIRED) == 1
    assert results.count(ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS) == 1
    assert len(reservations) == 2
    repository.close()


@pytest.mark.parametrize("iteration", range(3))
def test_same_key_race_winner_publishes_loser_waits(iteration: int) -> None:
    repository = _repository()
    key = _lookup_key()
    start_barrier = threading.Barrier(2)
    both_acquired = threading.Event()
    published = threading.Event()
    results: dict[str, ArtifactCreationCoordinationResult] = {}
    wait_outcomes: dict[str, bool] = {}
    lookups: dict[str, StoredOptimizationArtifact | None] = {}
    lock = threading.Lock()

    def contender(name: str) -> None:
        start_barrier.wait()
        result = repository.try_acquire_creation_reservation(
            key,
            owner_operation_id=name,
            lease_seconds=60,
        )
        with lock:
            results[name] = result
            if len(results) == 2:
                both_acquired.set()

        both_acquired.wait(timeout=5.0)

        if result.status is ArtifactCreationCoordinationStatus.ACQUIRED:
            if result.reservation is not None:
                _publish(
                    repository,
                    result.reservation,
                    key,
                    payload=b"winner-payload",
                    artifact_id="artifact-race",
                )
            published.set()
        else:
            published.wait(timeout=5.0)
            changed = repository.wait_for_artifact_or_reservation_change(
                key,
                observed_state_version=result.state_version,
                timeout_seconds=5.0,
            )
            with lock:
                wait_outcomes[name] = changed
                lookups[name] = repository.lookup(key)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(contender, "owner-a")
        executor.submit(contender, "owner-b")
        executor.shutdown(wait=True)

    statuses = [result.status for result in results.values()]
    assert statuses.count(ArtifactCreationCoordinationStatus.ACQUIRED) == 1
    assert statuses.count(ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS) == 1

    follower = next(
        name
        for name, result in results.items()
        if result.status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS
    )
    assert wait_outcomes[follower] is True
    lookup = lookups[follower]
    assert lookup is not None
    assert lookup.payload == b"winner-payload"
    repository.close()


def test_different_keys_both_acquire() -> None:
    repository = _repository()
    key_a = _lookup_key(source_refs=("msg-a",))
    key_b = _lookup_key(source_refs=("msg-b",))
    barrier = threading.Barrier(2)
    statuses: list[ArtifactCreationCoordinationStatus] = []
    lock = threading.Lock()

    def contender(key: ArtifactLookupKey, owner: str) -> None:
        barrier.wait()
        result = repository.try_acquire_creation_reservation(
            key,
            owner_operation_id=owner,
            lease_seconds=60,
        )
        with lock:
            statuses.append(result.status)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(contender, key_a, "owner-a")
        executor.submit(contender, key_b, "owner-b")
        executor.shutdown(wait=True)

    assert statuses == [
        ArtifactCreationCoordinationStatus.ACQUIRED,
        ArtifactCreationCoordinationStatus.ACQUIRED,
    ]
    repository.close()


def test_tenant_isolation_race_both_acquire() -> None:
    repository = _repository()
    key_a = _lookup_key(tenant_id="tenant-a")
    key_b = _lookup_key(tenant_id="tenant-b")
    barrier = threading.Barrier(2)
    statuses: list[ArtifactCreationCoordinationStatus] = []
    lock = threading.Lock()

    def contender(key: ArtifactLookupKey) -> None:
        barrier.wait()
        result = repository.try_acquire_creation_reservation(
            key,
            owner_operation_id="owner-1",
            lease_seconds=60,
        )
        with lock:
            statuses.append(result.status)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(contender, key_a)
        executor.submit(contender, key_b)
        executor.shutdown(wait=True)

    assert statuses == [
        ArtifactCreationCoordinationStatus.ACQUIRED,
        ArtifactCreationCoordinationStatus.ACQUIRED,
    ]
    repository.close()


def test_wait_lease_expiry_without_background_thread() -> None:
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
