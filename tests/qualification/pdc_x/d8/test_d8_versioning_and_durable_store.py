# © Artur Czarnecki. All rights reserved.

"""PDC-D8-R003 (append-only versions), R007 (durable store), R014 (restart recovery)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from intergrax.contracts.artifact_ref import (
    ArtifactDigestMismatchError,
    ArtifactRetentionClass,
    compute_artifact_digest,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.sqlite_artifact_store import SQLiteArtifactStore
from intergrax.runtime.nexus.artifacts.store_base import (
    ArtifactDeletionRefused,
    ArtifactStore,
    ArtifactVersionNotFound,
)

from ._d8_support import TENANT_A, TENANT_B, make_artifact, sqlite_store

pytestmark = pytest.mark.unit


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ArtifactStore:
    if request.param == "memory":
        return InMemoryArtifactStore()
    return sqlite_store(tmp_path)


# --------------------------------------------------------------------- R003


def test_put_appends_a_new_version_instead_of_overwriting(store: ArtifactStore) -> None:
    first = store.put(make_artifact(data=b"v-one"))
    second = store.put(make_artifact(data=b"v-two"))
    third = store.put(make_artifact(data=b"v-three"))
    assert (first.version, second.version, third.version) == (1, 2, 3)
    assert store.get(TENANT_A, "art-1").data == b"v-three"  # latest
    assert store.get_version(TENANT_A, "art-1", 1).data == b"v-one"  # history is intact
    assert store.get_version(TENANT_A, "art-1", 2).data == b"v-two"
    assert [a.version for a in store.list_versions(TENANT_A, "art-1")] == [1, 2, 3]
    assert [a.artifact_id for a in store.list_for_run(TENANT_A, "run-1")] == ["art-1"]


def test_unknown_version_is_reported_not_invented(store: ArtifactStore) -> None:
    store.put(make_artifact())
    with pytest.raises(ArtifactVersionNotFound):
        store.get_version(TENANT_A, "art-1", 9)


def test_versions_are_tenant_scoped(store: ArtifactStore) -> None:
    store.put(make_artifact(tenant_id=TENANT_A, data=b"a"))
    store.put(make_artifact(tenant_id=TENANT_B, data=b"b"))
    assert store.get(TENANT_A, "art-1").data == b"a"
    assert store.get(TENANT_B, "art-1").data == b"b"
    assert store.get(TENANT_B, "art-1").version == 1


def test_store_has_no_update_or_overwrite_api() -> None:
    public = {n for n in dir(ArtifactStore) if not n.startswith("_")}
    assert not {n for n in public if n.startswith(("update", "overwrite", "replace", "set_"))}


# --------------------------------------------------------------------- R007 / R014


def test_sqlite_store_is_durable_single_host(tmp_path: Path) -> None:
    assert sqlite_store(tmp_path).persistence_topology is PersistenceTopology.DURABLE_SINGLE_HOST


def test_artifacts_survive_a_restart(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    store.put(make_artifact(artifact_id="keep", data=b"v1"))
    store.put(make_artifact(artifact_id="keep", data=b"v2"))
    store.put(make_artifact(artifact_id="other", data=b"o"))
    store.place_legal_hold(TENANT_A, "other", placed_by="counsel", reason="case 42")
    del store  # process restart: a new store instance over the same file

    reopened = SQLiteArtifactStore(str(tmp_path / "artifacts.sqlite3"))
    latest = reopened.get(TENANT_A, "keep")
    assert latest.data == b"v2" and latest.version == 2
    assert reopened.get_version(TENANT_A, "keep", 1).data == b"v1"
    assert {a.artifact_id for a in reopened.list_for_run(TENANT_A, "run-1")} == {"keep", "other"}
    # content, digest, producer and lineage come back intact
    ref = latest.to_durable_ref()
    assert ref.content_digest == compute_artifact_digest(b"v2")
    assert ref.producer.created_by_agent_id == "agent-1"
    assert ref.lineage.task_id == "task-1"
    assert reopened.is_under_legal_hold(TENANT_A, "other")


def test_recovery_after_restart_continues_the_version_sequence(tmp_path: Path) -> None:
    first = sqlite_store(tmp_path)
    first.put(make_artifact(data=b"v1"))
    del first
    second = sqlite_store(tmp_path)
    appended = second.put(make_artifact(data=b"v2"))
    assert appended.version == 2
    third = sqlite_store(tmp_path)
    assert [a.version for a in third.list_versions(TENANT_A, "art-1")] == [1, 2]


def test_recovery_after_restart_protects_evidence_and_holds(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    store.put(make_artifact(artifact_id="ev", retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE))
    store.put(make_artifact(artifact_id="held"))
    store.place_legal_hold(TENANT_A, "held", placed_by="counsel", reason="r")
    del store
    reopened = sqlite_store(tmp_path)
    with pytest.raises(ArtifactDeletionRefused):
        reopened.delete_artifact(TENANT_A, "ev")
    with pytest.raises(ArtifactDeletionRefused):
        reopened.delete_artifact(TENANT_A, "held")
    assert reopened.get(TENANT_A, "ev").retention_class is ArtifactRetentionClass.AUDIT_EVIDENCE


def test_tampered_row_is_detected_on_recovery(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    store.put(make_artifact(data=b"original"))
    del store
    connection = sqlite3.connect(str(tmp_path / "artifacts.sqlite3"))
    try:
        connection.execute("UPDATE execution_artifacts SET data = ? WHERE artifact_id = 'art-1'", (b"forged",))
        connection.commit()
    finally:
        connection.close()
    reopened = sqlite_store(tmp_path)
    with pytest.raises(ArtifactDigestMismatchError):
        reopened.get(TENANT_A, "art-1")


def test_sqlite_rejects_overwriting_an_existing_version_row(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    store.put(make_artifact())
    connection = sqlite3.connect(str(tmp_path / "artifacts.sqlite3"))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO execution_artifacts (tenant_id, artifact_id, version, run_id, kind,"
                " mime_type, created_at_utc, data, size_bytes, content_digest, artifact_class,"
                " retention_class) VALUES ('tenant-a','art-1',1,'run-1','k','t','x',x'00',1,"
                " 'sha256:" + "0" * 64 + "','execution_evidence','default')"
            )
    finally:
        connection.close()


def test_cross_tenant_get_still_raises_key_error(store: ArtifactStore) -> None:
    store.put(make_artifact(tenant_id=TENANT_A))
    with pytest.raises(KeyError):
        store.get(TENANT_B, "art-1")
