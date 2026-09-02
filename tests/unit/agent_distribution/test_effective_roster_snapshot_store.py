# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-007 Phase 4A — effective roster snapshot authority store tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.errors import EffectiveRosterSnapshotConflict
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryEffectiveRosterSnapshotStore,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry

_DIGEST = "sha256:" + ("a" * 64)


def _entry(
    *,
    logical_agent_id: str = "search",
    installation_slot_id: str = "slot-search",
    active_installation_id: str = "inst-search-1",
    package_digest: str = _DIGEST,
    distribution_package_id: str = "intergrax-local-search-agent",
    effective_enablement: bool = True,
    merged_config: dict[str, object] | None = None,
    factory_reference: AgentBindingFactoryReference | None = None,
) -> EffectiveRosterEntry:
    return EffectiveRosterEntry(
        logical_agent_id=logical_agent_id,
        installation_slot_id=installation_slot_id,
        active_installation_id=active_installation_id,
        package_digest=package_digest,
        distribution_package_id=distribution_package_id,
        effective_enablement=effective_enablement,
        merged_config=merged_config
        or {"limits": {"rpm": 100}, "routing": {"mode": "balanced"}},
        factory_reference=factory_reference
        or AgentBindingFactoryReference(
            builder_key="search-builder",
            factory_path="agents.search.factory:build",
        ),
    )


def _roster(
    *,
    application_id: str = "demo_app",
    application_environment_id: str = "env-prod",
    manifest_release_id: str = "rel-1",
    binding_revisions: tuple[int, ...] = (1, 2),
    entry: EffectiveRosterEntry | None = None,
    finalize: bool = True,
) -> EffectiveRoster:
    roster = EffectiveRoster(
        application_id=application_id,
        application_environment_id=application_environment_id,
        manifest_release_id=manifest_release_id,
        binding_revisions=binding_revisions,
        entries=(entry or _entry(),),
    )
    return roster.with_revision_id() if finalize else roster


def test_valid_roster_snapshot_persist() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryEffectiveRosterSnapshotStore(state)
    roster = _roster()
    persisted = store.persist(roster)
    assert persisted == roster
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    assert store.get_by_revision(revision_id) == roster
    assert len(state.effective_roster_snapshots) == 1


def test_missing_lookup_returns_none() -> None:
    store = InMemoryEffectiveRosterSnapshotStore()
    assert store.get_by_revision("sha256:" + ("f" * 64)) is None


def test_exact_replay_idempotent() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryEffectiveRosterSnapshotStore(state)
    roster = _roster()
    first = store.persist(roster)
    second = store.persist(roster)
    assert second == first
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    assert store.get_by_revision(revision_id) == roster
    assert len(state.effective_roster_snapshots) == 1


def test_shared_state_visible_across_store_wrappers() -> None:
    state = AgentDistributionStoreState()
    store_a = InMemoryEffectiveRosterSnapshotStore(state)
    store_b = InMemoryEffectiveRosterSnapshotStore(state)
    roster = _roster()
    store_a.persist(roster)
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    assert store_b.get_by_revision(revision_id) == roster


def test_independent_states_isolated() -> None:
    state_a = AgentDistributionStoreState()
    state_b = AgentDistributionStoreState()
    store_a = InMemoryEffectiveRosterSnapshotStore(state_a)
    store_b = InMemoryEffectiveRosterSnapshotStore(state_b)
    roster = _roster()
    store_a.persist(roster)
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    assert store_b.get_by_revision(revision_id) is None


def test_multiple_different_roster_ids_coexist() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryEffectiveRosterSnapshotStore(state)
    first = _roster(application_id="app-a")
    second = _roster(application_id="app-b")
    store.persist(first)
    store.persist(second)
    first_id = first.effective_roster_revision_id
    second_id = second.effective_roster_revision_id
    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id
    assert store.get_by_revision(first_id) == first
    assert store.get_by_revision(second_id) == second
    assert len(state.effective_roster_snapshots) == 2


def test_roster_without_effective_roster_revision_id_rejected() -> None:
    store = InMemoryEffectiveRosterSnapshotStore()
    roster = _roster(finalize=False)
    with pytest.raises(
        EffectiveRosterSnapshotConflict,
        match="requires effective_roster_revision_id",
    ):
        store.persist(roster)


def test_roster_with_incorrect_embedded_revision_id_rejected() -> None:
    store = InMemoryEffectiveRosterSnapshotStore()
    roster = _roster().model_copy(
        update={"effective_roster_revision_id": "sha256:" + ("0" * 64)}
    )
    with pytest.raises(
        EffectiveRosterSnapshotConflict,
        match="does not match computed content hash",
    ):
        store.persist(roster)


def test_same_revision_id_different_payload_conflicts() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryEffectiveRosterSnapshotStore(state)
    roster_a = _roster()
    revision_id = roster_a.effective_roster_revision_id
    assert revision_id is not None
    roster_b = _roster(
        entry=_entry(distribution_package_id="intergrax-other-agent"),
    )
    state.effective_roster_snapshots[revision_id] = roster_b.model_copy(
        update={"effective_roster_revision_id": revision_id}
    )
    with pytest.raises(
        EffectiveRosterSnapshotConflict,
        match="authority conflict",
    ):
        store.persist(roster_a)


def test_corrupt_stored_payload_detected_on_read() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryEffectiveRosterSnapshotStore(state)
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    corrupt = roster.model_copy(
        update={"effective_roster_revision_id": "sha256:" + ("9" * 64)}
    )
    state.effective_roster_snapshots[revision_id] = corrupt
    with pytest.raises(
        EffectiveRosterSnapshotConflict,
        match="embedded revision id mismatch",
    ):
        store.get_by_revision(revision_id)


def test_requested_id_vs_embedded_id_mismatch_detected() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryEffectiveRosterSnapshotStore(state)
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    other_id = "sha256:" + ("c" * 64)
    assert other_id != revision_id
    state.effective_roster_snapshots[other_id] = roster
    with pytest.raises(
        EffectiveRosterSnapshotConflict,
        match="embedded revision id mismatch",
    ):
        store.get_by_revision(other_id)


def test_factory_reference_and_config_survive_roundtrip() -> None:
    store = InMemoryEffectiveRosterSnapshotStore()
    roster = _roster()
    persisted = store.persist(roster)
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    loaded = store.get_by_revision(revision_id)
    assert loaded == persisted
    entry = loaded.entries[0]
    assert entry.logical_agent_id == "search"
    assert entry.installation_slot_id == "slot-search"
    assert entry.active_installation_id == "inst-search-1"
    assert entry.package_digest == _DIGEST
    assert entry.distribution_package_id == "intergrax-local-search-agent"
    assert entry.effective_enablement is True
    assert entry.merged_config == {
        "limits": {"rpm": 100},
        "routing": {"mode": "balanced"},
    }
    assert entry.factory_reference == AgentBindingFactoryReference(
        builder_key="search-builder",
        factory_path="agents.search.factory:build",
    )


def test_snapshot_object_remains_frozen_through_store_use() -> None:
    store = InMemoryEffectiveRosterSnapshotStore()
    roster = _roster()
    store.persist(roster)
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    loaded = store.get_by_revision(revision_id)
    assert loaded is not None
    with pytest.raises(ValidationError):
        loaded.application_id = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        loaded.entries[0].merged_config["limits"]["rpm"] = 999


def test_concurrent_identical_persist_across_wrappers_idempotent() -> None:
    state = AgentDistributionStoreState()
    store_a = InMemoryEffectiveRosterSnapshotStore(state)
    store_b = InMemoryEffectiveRosterSnapshotStore(state)
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    barrier = threading.Barrier(2)
    results: list[EffectiveRoster] = []
    errors: list[Exception] = []

    def persist_from_a() -> None:
        barrier.wait(timeout=5)
        try:
            results.append(store_a.persist(roster))
        except Exception as exc:
            errors.append(exc)

    def persist_from_b() -> None:
        barrier.wait(timeout=5)
        try:
            results.append(store_b.persist(roster))
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(persist_from_a)
        future_b = executor.submit(persist_from_b)
        future_a.result(timeout=5)
        future_b.result(timeout=5)

    assert not errors
    assert len(results) == 2
    assert results[0] == roster
    assert results[1] == roster
    assert len(state.effective_roster_snapshots) == 1
    assert state.effective_roster_snapshots[revision_id] == roster


def test_corrupt_stored_payload_hash_failure_on_read() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryEffectiveRosterSnapshotStore(state)
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    corrupt_entry = roster.entries[0].model_copy(
        update={"distribution_package_id": "tampered-agent"}
    )
    corrupt = roster.model_copy(update={"entries": (corrupt_entry,)})
    state.effective_roster_snapshots[revision_id] = corrupt
    with pytest.raises(
        EffectiveRosterSnapshotConflict,
        match="content integrity failure on read",
    ):
        store.get_by_revision(revision_id)


def test_preseeded_conflict_blocks_overwrite() -> None:
    state = AgentDistributionStoreState()
    store_a = InMemoryEffectiveRosterSnapshotStore(state)
    store_b = InMemoryEffectiveRosterSnapshotStore(state)
    roster_a = _roster()
    revision_id = roster_a.effective_roster_revision_id
    assert revision_id is not None
    roster_b = _roster(
        entry=_entry(distribution_package_id="intergrax-other-agent"),
    )
    state.effective_roster_snapshots[revision_id] = roster_b.model_copy(
        update={"effective_roster_revision_id": revision_id}
    )
    with pytest.raises(
        EffectiveRosterSnapshotConflict,
        match="authority conflict",
    ):
        store_b.persist(roster_a)
    assert state.effective_roster_snapshots[revision_id] != roster_a
    with pytest.raises(EffectiveRosterSnapshotConflict):
        store_a.get_by_revision(revision_id)
