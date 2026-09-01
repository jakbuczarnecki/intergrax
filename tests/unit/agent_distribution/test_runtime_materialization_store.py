# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-006 Phase 1 — runtime materialization authority store tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.errors import RuntimeMaterializationConflict
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryRuntimeMaterializationStore,
)
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology

_ARTIFACT_DIGEST = "sha256:" + ("a" * 64)
_OTHER_ARTIFACT_DIGEST = "sha256:" + ("b" * 64)
_LOCK_ID = "sha256:" + ("c" * 64)
_OTHER_LOCK_ID = "sha256:" + ("d" * 64)
_OTHER_LOCK_DIGEST = "sha256:" + ("e" * 64)


def _record(
    *,
    runtime_revision_id: str = "rev-1",
    application_id: str = "app-a",
    application_environment_id: str = "prod",
    materialization_topology: MaterializationTopology = MaterializationTopology.OCI_IMAGE,
    artifact_locator: str = "file:///artifact/a",
    materialization_artifact_digest: str = _ARTIFACT_DIGEST,
    materialized_runtime_lock_id: str = _LOCK_ID,
    materialized_runtime_lock_digest: str = _LOCK_ID,
) -> RuntimeMaterializationRecord:
    return RuntimeMaterializationRecord(
        runtime_revision_id=runtime_revision_id,
        application_id=application_id,
        application_environment_id=application_environment_id,
        materialization_topology=materialization_topology,
        artifact_locator=artifact_locator,
        materialization_artifact_digest=materialization_artifact_digest,
        materialized_runtime_lock_id=materialized_runtime_lock_id,
        materialized_runtime_lock_digest=materialized_runtime_lock_digest,
    )


def test_record_construct_valid() -> None:
    record = _record()
    assert record.runtime_revision_id == "rev-1"
    assert record.artifact_locator == "file:///artifact/a"


def test_record_frozen() -> None:
    record = _record()
    with pytest.raises(ValidationError):
        record.application_id = "app-b"  # type: ignore[misc]


def test_record_rejects_blank_required_field() -> None:
    with pytest.raises(ValidationError):
        _record(application_id="   ")


def test_record_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        RuntimeMaterializationRecord.model_validate(
            {
                **_record().model_dump(),
                "unexpected": "value",
            }
        )


def test_store_missing_lookup_returns_none() -> None:
    store = InMemoryRuntimeMaterializationStore()
    assert store.get_by_revision("rev-missing") is None


def test_first_persist() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryRuntimeMaterializationStore(state)
    record = _record()
    persisted = store.persist(record)
    assert persisted == record
    assert store.get_by_revision("rev-1") == record
    assert len(state.materializations) == 1


def test_exact_replay_idempotent() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryRuntimeMaterializationStore(state)
    record = _record()
    first = store.persist(record)
    second = store.persist(record)
    assert second == first
    assert store.get_by_revision("rev-1") == record
    assert len(state.materializations) == 1


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("application_id", "app-b"),
        ("application_environment_id", "staging"),
        ("artifact_locator", "file:///artifact/b"),
        ("materialization_artifact_digest", _OTHER_ARTIFACT_DIGEST),
        ("materialization_topology", MaterializationTopology.VENV_BUNDLE),
        ("materialized_runtime_lock_id", _OTHER_LOCK_ID),
        ("materialized_runtime_lock_digest", _OTHER_LOCK_DIGEST),
    ],
)
def test_same_revision_different_authority_conflicts(
    field_name: str,
    field_value: object,
) -> None:
    store = InMemoryRuntimeMaterializationStore()
    store.persist(_record())
    conflict = _record(**{field_name: field_value})
    with pytest.raises(RuntimeMaterializationConflict, match="authority conflict"):
        store.persist(conflict)


def test_locator_conflict_even_when_digest_identical() -> None:
    store = InMemoryRuntimeMaterializationStore()
    store.persist(_record(artifact_locator="file:///artifact/a"))
    with pytest.raises(RuntimeMaterializationConflict):
        store.persist(_record(artifact_locator="file:///artifact/b"))


def test_shared_state_visible_across_store_wrappers() -> None:
    state = AgentDistributionStoreState()
    store_a = InMemoryRuntimeMaterializationStore(state)
    store_b = InMemoryRuntimeMaterializationStore(state)
    record = _record()
    store_a.persist(record)
    assert store_b.get_by_revision("rev-1") == record


def test_independent_states_isolated() -> None:
    state_a = AgentDistributionStoreState()
    state_b = AgentDistributionStoreState()
    store_a = InMemoryRuntimeMaterializationStore(state_a)
    store_b = InMemoryRuntimeMaterializationStore(state_b)
    store_a.persist(_record())
    assert store_b.get_by_revision("rev-1") is None


def test_two_different_revision_ids_can_coexist() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryRuntimeMaterializationStore(state)
    first = _record(runtime_revision_id="rev-1")
    second = _record(runtime_revision_id="rev-2")
    store.persist(first)
    store.persist(second)
    assert store.get_by_revision("rev-1") == first
    assert store.get_by_revision("rev-2") == second
    assert len(state.materializations) == 2


def test_test_scheme_locator_accepted_as_opaque() -> None:
    store = InMemoryRuntimeMaterializationStore()
    record = _record(artifact_locator="test://opaque/locator")
    persisted = store.persist(record)
    assert persisted.artifact_locator == "test://opaque/locator"


def test_concurrent_conflicting_persist_across_wrappers_one_wins() -> None:
    state = AgentDistributionStoreState()
    store_a = InMemoryRuntimeMaterializationStore(state)
    store_b = InMemoryRuntimeMaterializationStore(state)
    record_a = _record(artifact_locator="file:///artifact/a")
    record_b = _record(artifact_locator="file:///artifact/b")
    barrier = threading.Barrier(2)
    successes: list[RuntimeMaterializationRecord] = []
    conflicts: list[RuntimeMaterializationConflict] = []

    def persist_from_a() -> None:
        barrier.wait(timeout=5)
        try:
            successes.append(store_a.persist(record_a))
        except RuntimeMaterializationConflict as exc:
            conflicts.append(exc)

    def persist_from_b() -> None:
        barrier.wait(timeout=5)
        try:
            successes.append(store_b.persist(record_b))
        except RuntimeMaterializationConflict as exc:
            conflicts.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(persist_from_a)
        future_b = executor.submit(persist_from_b)
        future_a.result(timeout=5)
        future_b.result(timeout=5)

    assert len(successes) == 1
    assert len(conflicts) == 1
    canonical = successes[0]
    assert canonical in {record_a, record_b}
    assert len(state.materializations) == 1
    assert state.materializations["rev-1"] == canonical
    assert store_a.get_by_revision("rev-1") == canonical
    assert store_b.get_by_revision("rev-1") == canonical


def test_concurrent_identical_persist_across_wrappers_idempotent() -> None:
    state = AgentDistributionStoreState()
    store_a = InMemoryRuntimeMaterializationStore(state)
    store_b = InMemoryRuntimeMaterializationStore(state)
    record = _record()
    barrier = threading.Barrier(2)
    results: list[RuntimeMaterializationRecord] = []
    errors: list[Exception] = []

    def persist_from_a() -> None:
        barrier.wait(timeout=5)
        try:
            results.append(store_a.persist(record))
        except Exception as exc:
            errors.append(exc)

    def persist_from_b() -> None:
        barrier.wait(timeout=5)
        try:
            results.append(store_b.persist(record))
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(persist_from_a)
        future_b = executor.submit(persist_from_b)
        future_a.result(timeout=5)
        future_b.result(timeout=5)

    assert not errors
    assert len(results) == 2
    assert results[0] == record
    assert results[1] == record
    assert len(state.materializations) == 1
    assert state.materializations["rev-1"] == record


@pytest.mark.parametrize("run_index", range(30))
def test_concurrent_conflicting_persist_across_wrappers_repeated(
    run_index: int,
) -> None:
    del run_index
    test_concurrent_conflicting_persist_across_wrappers_one_wins()
