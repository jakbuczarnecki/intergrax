# © Artur Czarnecki. All rights reserved.

"""MP-3D — SQLite WorkArtifact repository semantic and durability tests."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositoriesWithArtifacts,
    open_sqlite_collaborative_work_repositories,
)
from intergrax.collaborative_work.repository import (
    ArtifactPublicationIdempotencyConflict,
    ArtifactPublicationRepository,
    CreateArtifactWithInitialVersionCommand,
    INITIAL_RECORD_REVISION,
    PublishWorkArtifactVersionCommand,
    PublishedWorkArtifactVersion,
    WorkArtifactAlreadyExists,
    WorkArtifactIdempotencyConflict,
    WorkArtifactNotFound,
    WorkArtifactRepository,
    WorkArtifactRevisionConflict,
    WorkArtifactTemporalConflict,
    WorkArtifactVersionAlreadyExists,
    WorkArtifactVersionRepository,
)
from intergrax.collaborative_work.serialization import (
    published_work_artifact_version_from_json,
    published_work_artifact_version_to_json,
    work_artifact_version_from_json,
    work_artifact_version_to_json,
)
from intergrax.collaborative_work.sqlite_repository import SQLiteArtifactPublicationRepository
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    CollaborativeWorkArtifactInvariantError,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE_A = "workspace-a"
_WORKSPACE_B = "workspace-b"
_WORK_ITEM = "work-item-1"
_ARTIFACT = "artifact-1"
_VERSION_1 = "artifact-version-1"
_VERSION_2 = "artifact-version-2"
_VERSION_3 = "artifact-version-3"
_DIGEST = "sha256:" + ("a" * 64)
_CREATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_UPDATED_AT = _CREATED_AT + timedelta(minutes=1)
_PUBLISHED_AT = _CREATED_AT + timedelta(minutes=5)
_LATER_PUBLISHED = _PUBLISHED_AT + timedelta(minutes=5)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "artifact.sqlite")


@pytest.fixture
def bundle(db_path: str) -> CollaborativeWorkRepositoriesWithArtifacts:
    opened = open_sqlite_collaborative_work_repositories(db_path)
    yield opened
    opened.close()


@pytest.fixture
def artifact_repo(bundle: CollaborativeWorkRepositoriesWithArtifacts) -> WorkArtifactRepository:
    return bundle.artifact


@pytest.fixture
def version_repo(bundle: CollaborativeWorkRepositoriesWithArtifacts) -> WorkArtifactVersionRepository:
    return bundle.version


@pytest.fixture
def publication_repo(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> ArtifactPublicationRepository:
    return bundle.publication


def _content_ref(**overrides: object) -> ArtifactContentRef:
    payload = {
        "content_ref": "content://tenant-a/workspace-a/body-1",
        "media_type": "application/json",
        "integrity_digest": _DIGEST,
    }
    payload.update(overrides)
    return ArtifactContentRef.model_validate(payload)


def _execution(**overrides: object) -> ExecutionProvenanceRef:
    payload = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
    }
    payload.update(overrides)
    return ExecutionProvenanceRef(**payload)


def _create_command(**overrides: object) -> CreateArtifactWithInitialVersionCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "work_item_id": _WORK_ITEM,
        "work_artifact_id": _ARTIFACT,
        "work_artifact_version_id": _VERSION_1,
        "created_by_principal_id": "principal-creator",
        "published_by_principal_id": "principal-publisher",
        "content_ref": _content_ref(),
        "artifact_created_at": _CREATED_AT,
        "artifact_updated_at": _UPDATED_AT,
        "version_created_at": _CREATED_AT,
        "version_published_at": _PUBLISHED_AT,
        "execution": None,
    }
    payload.update(overrides)
    return CreateArtifactWithInitialVersionCommand(**payload)


def _publish_command(**overrides: object) -> PublishWorkArtifactVersionCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "work_item_id": _WORK_ITEM,
        "work_artifact_id": _ARTIFACT,
        "work_artifact_version_id": _VERSION_2,
        "expected_revision": 0,
        "created_by_principal_id": "principal-creator",
        "published_by_principal_id": "principal-publisher",
        "content_ref": _content_ref(content_ref="content://tenant-a/workspace-a/body-2"),
        "created_at": _UPDATED_AT,
        "published_at": _LATER_PUBLISHED,
        "artifact_updated_at": _LATER_PUBLISHED,
        "execution": None,
    }
    payload.update(overrides)
    return PublishWorkArtifactVersionCommand(**payload)


def _seed_initial(publication_repo: ArtifactPublicationRepository) -> PublishedWorkArtifactVersion:
    return publication_repo.create_artifact_with_initial_version(_create_command())


def _reopen(db_path: str) -> CollaborativeWorkRepositoriesWithArtifacts:
    return open_sqlite_collaborative_work_repositories(db_path)


def test_repository_protocols_are_satisfied(db_path: str) -> None:
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    try:
        assert isinstance(bundle, CollaborativeWorkRepositoriesWithArtifacts)
        assert isinstance(bundle.artifact, WorkArtifactRepository)
        assert isinstance(bundle.version, WorkArtifactVersionRepository)
        assert isinstance(bundle.publication, ArtifactPublicationRepository)
        assert bundle.artifact is bundle.artifacts.artifact
        assert bundle.core.store is bundle.store
    finally:
        bundle.close()


def test_initial_create_sets_revision_and_pointer(
    publication_repo: ArtifactPublicationRepository,
    artifact_repo: WorkArtifactRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    result = publication_repo.create_artifact_with_initial_version(_create_command())
    assert result.artifact.revision == INITIAL_RECORD_REVISION
    assert result.artifact.current_version_id == _VERSION_1
    loaded_artifact = artifact_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    loaded_version = version_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_version_id=_VERSION_1,
    )
    assert loaded_artifact == result.artifact
    assert loaded_version == result.version


def test_duplicate_artifact_identity_raises(publication_repo: ArtifactPublicationRepository) -> None:
    publication_repo.create_artifact_with_initial_version(_create_command())
    with pytest.raises(WorkArtifactAlreadyExists):
        publication_repo.create_artifact_with_initial_version(
            _create_command(work_artifact_version_id="artifact-version-other"),
        )


def test_duplicate_version_identity_raises(publication_repo: ArtifactPublicationRepository) -> None:
    publication_repo.create_artifact_with_initial_version(_create_command())
    with pytest.raises(WorkArtifactVersionAlreadyExists):
        publication_repo.create_artifact_with_initial_version(
            _create_command(work_artifact_id="artifact-other"),
        )


def test_initial_create_idempotency_replay_and_conflict(
    publication_repo: ArtifactPublicationRepository,
) -> None:
    command = _create_command(idempotency_key="create-idem")
    created = publication_repo.create_artifact_with_initial_version(command)
    replay = publication_repo.create_artifact_with_initial_version(command)
    assert replay == created
    with pytest.raises(WorkArtifactIdempotencyConflict):
        publication_repo.create_artifact_with_initial_version(
            _create_command(work_artifact_id="artifact-other", idempotency_key="create-idem"),
        )


def test_initial_create_idempotency_preserves_timestamps(
    publication_repo: ArtifactPublicationRepository,
) -> None:
    first_command = _create_command(
        idempotency_key="create-idem",
        artifact_created_at=_CREATED_AT,
        artifact_updated_at=_UPDATED_AT,
        version_created_at=_CREATED_AT,
        version_published_at=_PUBLISHED_AT,
    )
    created = publication_repo.create_artifact_with_initial_version(first_command)
    replay = publication_repo.create_artifact_with_initial_version(
        _create_command(
            idempotency_key="create-idem",
            artifact_created_at=_LATER_PUBLISHED,
            artifact_updated_at=_LATER_PUBLISHED + timedelta(minutes=1),
            version_created_at=_LATER_PUBLISHED,
            version_published_at=_LATER_PUBLISHED + timedelta(minutes=2),
        ),
    )
    assert replay == created
    assert replay.artifact.created_at == created.artifact.created_at
    assert replay.version.published_at == created.version.published_at


def test_initial_create_failure_leaves_no_partial_state(
    publication_repo: SQLiteArtifactPublicationRepository,
    db_path: str,
) -> None:
    with patch(
        "intergrax.collaborative_work.sqlite_repository.validate_work_artifact_current_version",
        side_effect=CollaborativeWorkArtifactInvariantError("forced failure"),
    ):
        with pytest.raises(CollaborativeWorkArtifactInvariantError):
            publication_repo.create_artifact_with_initial_version(_create_command())
    reopened = _reopen(db_path)
    try:
        assert (
            reopened.artifact.get(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                work_artifact_id=_ARTIFACT,
            )
            is None
        )
        assert (
            reopened.version.get(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                work_artifact_version_id=_VERSION_1,
            )
            is None
        )
        connection = sqlite3.connect(db_path)
        row = connection.execute(
            "SELECT COUNT(*) FROM collaborative_idempotency WHERE entity_kind = ?",
            ("artifact.create",),
        ).fetchone()
        connection.close()
        assert row is not None and int(row[0]) == 0
    finally:
        reopened.close()


def test_initial_create_restart_durability(db_path: str) -> None:
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    command = _create_command(idempotency_key="restart-create")
    created = bundle.publication.create_artifact_with_initial_version(command)
    bundle.close()

    reopened = _reopen(db_path)
    try:
        loaded_artifact = reopened.artifact.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        loaded_version = reopened.version.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=_VERSION_1,
        )
        assert loaded_artifact == created.artifact
        assert loaded_version == created.version
        replay = reopened.publication.create_artifact_with_initial_version(
            _create_command(
                idempotency_key="restart-create",
                artifact_created_at=_LATER_PUBLISHED,
                artifact_updated_at=_LATER_PUBLISHED,
                version_created_at=_LATER_PUBLISHED,
                version_published_at=_LATER_PUBLISHED,
            ),
        )
        assert replay == created
    finally:
        reopened.close()


def test_publish_advances_revision_and_preserves_history(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    initial = _seed_initial(publication_repo)
    published = publication_repo.publish_version(_publish_command())
    assert published.artifact.revision == initial.artifact.revision + 1
    listed = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert listed == (initial.version, published.version)


def test_publish_temporal_regression_rejected(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    initial = _seed_initial(publication_repo)
    with pytest.raises(WorkArtifactTemporalConflict):
        publication_repo.publish_version(
            _publish_command(
                artifact_updated_at=initial.artifact.updated_at - timedelta(seconds=1),
            ),
        )
    assert (
        version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=_VERSION_2,
        )
        is None
    )


def test_publish_temporal_equality_accepted(publication_repo: ArtifactPublicationRepository) -> None:
    initial = _seed_initial(publication_repo)
    published = publication_repo.publish_version(
        _publish_command(artifact_updated_at=initial.artifact.updated_at),
    )
    assert published.artifact.updated_at == initial.artifact.updated_at


def test_publish_stale_revision_conflicts(publication_repo: ArtifactPublicationRepository) -> None:
    _seed_initial(publication_repo)
    publication_repo.publish_version(_publish_command(expected_revision=0))
    with pytest.raises(WorkArtifactRevisionConflict):
        publication_repo.publish_version(
            _publish_command(work_artifact_version_id=_VERSION_3, expected_revision=0),
        )


def test_publish_wrong_scope_not_found(publication_repo: ArtifactPublicationRepository) -> None:
    _seed_initial(publication_repo)
    with pytest.raises(WorkArtifactNotFound):
        publication_repo.publish_version(
            _publish_command(tenant_id=_TENANT_B, workspace_id=_WORKSPACE_B),
        )


def test_publish_idempotency_replay_after_later_publication(
    publication_repo: ArtifactPublicationRepository,
) -> None:
    _seed_initial(publication_repo)
    first_publish = publication_repo.publish_version(_publish_command(idempotency_key="publish-idem"))
    publication_repo.publish_version(
        _publish_command(
            work_artifact_version_id=_VERSION_3,
            expected_revision=1,
            published_at=_LATER_PUBLISHED + timedelta(minutes=10),
            artifact_updated_at=_LATER_PUBLISHED + timedelta(minutes=10),
        ),
    )
    replay = publication_repo.publish_version(_publish_command(idempotency_key="publish-idem"))
    assert replay == first_publish
    assert replay.artifact.revision == 1


def test_publish_failure_leaves_no_partial_state(
    publication_repo: SQLiteArtifactPublicationRepository,
    artifact_repo: WorkArtifactRepository,
    db_path: str,
) -> None:
    initial = _seed_initial(publication_repo)
    with patch.object(
        publication_repo,
        "_store_publication_idempotency",
        side_effect=RuntimeError("forced failure"),
    ):
        with pytest.raises(RuntimeError):
            publication_repo.publish_version(_publish_command(idempotency_key="publish-fail"))
    unchanged = artifact_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert unchanged == initial.artifact
    reopened = _reopen(db_path)
    try:
        assert (
            reopened.version.get(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                work_artifact_version_id=_VERSION_2,
            )
            is None
        )
    finally:
        reopened.close()


def test_publish_restart_and_idempotency_replay(db_path: str) -> None:
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    bundle.publication.create_artifact_with_initial_version(_create_command())
    published = bundle.publication.publish_version(_publish_command(idempotency_key="restart-publish"))
    bundle.close()

    reopened = _reopen(db_path)
    try:
        replay = reopened.publication.publish_version(
            _publish_command(
                idempotency_key="restart-publish",
                published_at=_LATER_PUBLISHED + timedelta(hours=1),
                artifact_updated_at=_LATER_PUBLISHED + timedelta(hours=1),
            ),
        )
        assert replay == published
        reopened.publication.publish_version(
            _publish_command(
                work_artifact_version_id=_VERSION_3,
                expected_revision=1,
                published_at=_LATER_PUBLISHED + timedelta(minutes=10),
                artifact_updated_at=_LATER_PUBLISHED + timedelta(minutes=10),
            ),
        )
    finally:
        reopened.close()

    final = _reopen(db_path)
    try:
        old_replay = final.publication.publish_version(_publish_command(idempotency_key="restart-publish"))
        assert old_replay == published
    finally:
        final.close()


def test_version_history_survives_restart(db_path: str) -> None:
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    initial = bundle.publication.create_artifact_with_initial_version(_create_command())
    v1_json = work_artifact_version_to_json(initial.version)
    bundle.publication.publish_version(_publish_command())
    bundle.publication.publish_version(
        _publish_command(
            work_artifact_version_id=_VERSION_3,
            expected_revision=1,
            published_at=_LATER_PUBLISHED + timedelta(minutes=1),
            artifact_updated_at=_LATER_PUBLISHED + timedelta(minutes=1),
        ),
    )
    bundle.close()

    reopened = _reopen(db_path)
    try:
        listed = reopened.version.list_for_artifact(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        assert len(listed) == 3
        assert work_artifact_version_to_json(listed[0]) == v1_json
    finally:
        reopened.close()


def test_read_ports_scope_isolation(
    artifact_repo: WorkArtifactRepository,
    version_repo: WorkArtifactVersionRepository,
    publication_repo: ArtifactPublicationRepository,
) -> None:
    created = publication_repo.create_artifact_with_initial_version(_create_command())
    assert (
        artifact_repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        is None
    )
    assert (
        version_repo.list_for_artifact(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        == ()
    )
    with pytest.raises(ValidationError):
        created.artifact.revision = 99  # type: ignore[misc]


def test_concurrent_publish_one_wins(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    initial = _seed_initial(publication_repo)
    errors: list[BaseException] = []
    results: list[PublishedWorkArtifactVersion] = []
    barrier = threading.Barrier(2)

    def attempt(version_id: str) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                publication_repo.publish_version(
                    _publish_command(work_artifact_version_id=version_id, expected_revision=0),
                ),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=attempt, args=(_VERSION_2,)),
        threading.Thread(target=attempt, args=(_VERSION_3,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], WorkArtifactRevisionConflict)
    winner = results[0]
    history = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert history == (initial.version, winner.version)


def test_two_bundles_same_file_no_lost_update(db_path: str) -> None:
    bundle_a = open_sqlite_collaborative_work_repositories(db_path)
    bundle_b = open_sqlite_collaborative_work_repositories(db_path)
    try:
        initial = bundle_a.publication.create_artifact_with_initial_version(_create_command())
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def attempt(bundle: CollaborativeWorkRepositoriesWithArtifacts, version_id: str) -> None:
            try:
                barrier.wait(timeout=5)
                bundle.publication.publish_version(
                    _publish_command(
                        work_artifact_version_id=version_id,
                        expected_revision=initial.artifact.revision,
                    ),
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=attempt, args=(bundle_a, _VERSION_2)),
            threading.Thread(target=attempt, args=(bundle_b, _VERSION_3)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 1
        assert isinstance(errors[0], WorkArtifactRevisionConflict)
        final = bundle_a.artifact.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        assert final is not None
        assert final.revision == initial.artifact.revision + 1
    finally:
        bundle_a.close()
        bundle_b.close()


def test_legacy_database_opens_and_artifact_tables_usable(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-mp2.sqlite"
    connection = sqlite3.connect(str(db_path))
    connection.executescript(
        """
        CREATE TABLE work_items (
            tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            work_item_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            revision INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, workspace_id, work_item_id)
        );
        CREATE TABLE collaborative_idempotency (
            tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            entity_kind TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            semantic_fingerprint TEXT NOT NULL,
            result_json TEXT NOT NULL,
            PRIMARY KEY (tenant_id, workspace_id, entity_kind, idempotency_key)
        );
        """
    )
    connection.commit()
    connection.close()

    bundle = open_sqlite_collaborative_work_repositories(str(db_path))
    try:
        created = bundle.publication.create_artifact_with_initial_version(_create_command())
        loaded = bundle.artifact.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        assert loaded == created.artifact
    finally:
        bundle.close()


def test_publication_result_serialization_round_trip() -> None:
    artifact_bundle = open_sqlite_collaborative_work_repositories(":memory:")
    try:
        result = artifact_bundle.publication.create_artifact_with_initial_version(_create_command())
    finally:
        artifact_bundle.close()
    encoded = published_work_artifact_version_to_json(result)
    decoded = published_work_artifact_version_from_json(encoded)
    assert decoded == result


def test_work_artifact_version_from_json_rejects_unknown_top_level_field(
    publication_repo: ArtifactPublicationRepository,
) -> None:
    created = publication_repo.create_artifact_with_initial_version(_create_command())
    payload = json.loads(work_artifact_version_to_json(created.version))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        work_artifact_version_from_json(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def test_published_work_artifact_version_from_json_rejects_unknown_top_level_field(
    publication_repo: ArtifactPublicationRepository,
) -> None:
    created = publication_repo.create_artifact_with_initial_version(_create_command())
    payload = json.loads(published_work_artifact_version_to_json(created))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        published_work_artifact_version_from_json(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )


def test_published_work_artifact_version_from_json_rejects_nested_extra_field(
    publication_repo: ArtifactPublicationRepository,
) -> None:
    created = publication_repo.create_artifact_with_initial_version(_create_command())
    payload = json.loads(published_work_artifact_version_to_json(created))
    payload["version"]["unexpected"] = True
    with pytest.raises(ValidationError):
        published_work_artifact_version_from_json(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )


def test_execution_provenance_sqlite_round_trip_and_strict_read(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
    db_path: str,
) -> None:
    execution = _execution()
    created = publication_repo.create_artifact_with_initial_version(_create_command(execution=execution))
    loaded = version_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_version_id=_VERSION_1,
    )
    assert loaded == created.version
    assert loaded is not None and loaded.execution == execution

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        """
        SELECT record_json FROM work_artifact_versions
        WHERE tenant_id = ? AND workspace_id = ? AND work_artifact_version_id = ?
        """,
        (_TENANT_A, _WORKSPACE_A, _VERSION_1),
    ).fetchone()
    assert row is not None
    corrupted = json.loads(row[0])
    assert corrupted["execution"] is not None
    corrupted["execution"]["unexpected"] = "value"
    connection.execute(
        """
        UPDATE work_artifact_versions
        SET record_json = ?
        WHERE tenant_id = ? AND workspace_id = ? AND work_artifact_version_id = ?
        """,
        (
            json.dumps(corrupted, sort_keys=True, separators=(",", ":")),
            _TENANT_A,
            _WORKSPACE_A,
            _VERSION_1,
        ),
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValidationError):
        version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=_VERSION_1,
        )


def test_sqlite_corrupt_version_record_json_fails_closed_on_get(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
    db_path: str,
) -> None:
    publication_repo.create_artifact_with_initial_version(_create_command())

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        """
        SELECT record_json FROM work_artifact_versions
        WHERE tenant_id = ? AND workspace_id = ? AND work_artifact_version_id = ?
        """,
        (_TENANT_A, _WORKSPACE_A, _VERSION_1),
    ).fetchone()
    assert row is not None
    corrupted = json.loads(row[0])
    corrupted["unexpected"] = True
    connection.execute(
        """
        UPDATE work_artifact_versions
        SET record_json = ?
        WHERE tenant_id = ? AND workspace_id = ? AND work_artifact_version_id = ?
        """,
        (
            json.dumps(corrupted, sort_keys=True, separators=(",", ":")),
            _TENANT_A,
            _WORKSPACE_A,
            _VERSION_1,
        ),
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValidationError):
        version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=_VERSION_1,
        )


def test_sqlite_corrupt_idempotency_result_json_fails_closed_on_replay(
    publication_repo: ArtifactPublicationRepository,
    db_path: str,
) -> None:
    command = _create_command(idempotency_key="create-idem")
    publication_repo.create_artifact_with_initial_version(command)

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        """
        SELECT result_json FROM collaborative_idempotency
        WHERE tenant_id = ? AND workspace_id = ? AND entity_kind = ? AND idempotency_key = ?
        """,
        (_TENANT_A, _WORKSPACE_A, "artifact.create", "create-idem"),
    ).fetchone()
    assert row is not None
    corrupted = json.loads(row[0])
    corrupted["unexpected"] = True
    connection.execute(
        """
        UPDATE collaborative_idempotency
        SET result_json = ?
        WHERE tenant_id = ? AND workspace_id = ? AND entity_kind = ? AND idempotency_key = ?
        """,
        (
            json.dumps(corrupted, sort_keys=True, separators=(",", ":")),
            _TENANT_A,
            _WORKSPACE_A,
            "artifact.create",
            "create-idem",
        ),
    )
    connection.commit()
    connection.close()

    reopened = _reopen(db_path)
    try:
        with pytest.raises(ValidationError):
            reopened.publication.create_artifact_with_initial_version(command)
    finally:
        reopened.close()


def test_sqlite_corrupt_publish_idempotency_nested_field_fails_closed_on_replay(
    publication_repo: ArtifactPublicationRepository,
    db_path: str,
) -> None:
    _seed_initial(publication_repo)
    command = _publish_command(idempotency_key="publish-idem")
    publication_repo.publish_version(command)

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        """
        SELECT result_json FROM collaborative_idempotency
        WHERE tenant_id = ? AND workspace_id = ? AND entity_kind = ? AND idempotency_key = ?
        """,
        (_TENANT_A, _WORKSPACE_A, "artifact.publish", "publish-idem"),
    ).fetchone()
    assert row is not None
    corrupted = json.loads(row[0])
    corrupted["artifact"]["unexpected"] = True
    connection.execute(
        """
        UPDATE collaborative_idempotency
        SET result_json = ?
        WHERE tenant_id = ? AND workspace_id = ? AND entity_kind = ? AND idempotency_key = ?
        """,
        (
            json.dumps(corrupted, sort_keys=True, separators=(",", ":")),
            _TENANT_A,
            _WORKSPACE_A,
            "artifact.publish",
            "publish-idem",
        ),
    )
    connection.commit()
    connection.close()

    reopened = _reopen(db_path)
    try:
        with pytest.raises(ValidationError):
            reopened.publication.publish_version(command)
    finally:
        reopened.close()


def test_publish_idempotency_conflict(publication_repo: ArtifactPublicationRepository) -> None:
    _seed_initial(publication_repo)
    publication_repo.publish_version(_publish_command(idempotency_key="publish-idem"))
    with pytest.raises(ArtifactPublicationIdempotencyConflict):
        publication_repo.publish_version(
            _publish_command(work_artifact_version_id=_VERSION_3, idempotency_key="publish-idem"),
        )


def test_sqlite_bundle_close_semantics(db_path: str) -> None:
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    bundle.close()
    bundle.close()
    with pytest.raises(RuntimeError, match="closed"):
        bundle.artifact.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
