# © Artur Czarnecki. All rights reserved.

"""MP-3B — in-memory WorkArtifact repository semantic tests."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryArtifactPublicationRepository,
    InMemoryWorkArtifactRepository,
    InMemoryWorkArtifactVersionRepository,
    open_in_memory_artifact_repositories,
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
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    CollaborativeWorkArtifactInvariantError,
    validate_work_artifact_current_version,
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
def artifact_bundle():
    return open_in_memory_artifact_repositories()


@pytest.fixture
def artifact_repo(artifact_bundle):
    return artifact_bundle.artifact


@pytest.fixture
def version_repo(artifact_bundle):
    return artifact_bundle.version


@pytest.fixture
def publication_repo(artifact_bundle):
    return artifact_bundle.publication


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


def test_repository_protocols_are_satisfied() -> None:
    bundle = open_in_memory_artifact_repositories()
    assert isinstance(bundle.artifact, WorkArtifactRepository)
    assert isinstance(bundle.version, WorkArtifactVersionRepository)
    assert isinstance(bundle.publication, ArtifactPublicationRepository)


def test_initial_create_sets_revision_and_pointer(
    publication_repo: ArtifactPublicationRepository,
    artifact_repo: WorkArtifactRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    result = publication_repo.create_artifact_with_initial_version(_create_command())
    assert result.artifact.revision == INITIAL_RECORD_REVISION
    assert result.artifact.current_version_id == _VERSION_1
    assert result.version.work_artifact_version_id == _VERSION_1
    assert result.artifact.tenant_id == _TENANT_A
    assert result.version.tenant_id == _TENANT_A

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
    listed = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert listed == (result.version,)


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


def test_cross_tenant_and_workspace_isolation(publication_repo: ArtifactPublicationRepository) -> None:
    publication_repo.create_artifact_with_initial_version(_create_command())
    publication_repo.create_artifact_with_initial_version(
        _create_command(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            work_artifact_id=_ARTIFACT,
            work_artifact_version_id=_VERSION_1,
        ),
    )
    publication_repo.create_artifact_with_initial_version(
        _create_command(
            workspace_id=_WORKSPACE_B,
            work_artifact_id="artifact-ws-b",
            work_artifact_version_id="artifact-version-ws-b",
        ),
    )


def test_initial_create_idempotency_replay_and_conflict(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    command = _create_command(idempotency_key="create-idem")
    created = publication_repo.create_artifact_with_initial_version(command)
    replay = publication_repo.create_artifact_with_initial_version(command)
    assert replay == created
    assert replay.artifact.updated_at == created.artifact.updated_at
    assert (
        version_repo.list_for_artifact(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        == (created.version,)
    )
    with pytest.raises(WorkArtifactIdempotencyConflict):
        publication_repo.create_artifact_with_initial_version(
            _create_command(
                work_artifact_id="artifact-other",
                idempotency_key="create-idem",
            ),
        )


def test_initial_create_failure_leaves_no_partial_state(
    publication_repo: InMemoryArtifactPublicationRepository,
) -> None:
    store = publication_repo._store
    with patch(
        "intergrax.collaborative_work.in_memory_repository.validate_work_artifact_current_version",
        side_effect=CollaborativeWorkArtifactInvariantError("forced failure"),
    ):
        with pytest.raises(CollaborativeWorkArtifactInvariantError):
            publication_repo.create_artifact_with_initial_version(_create_command())
    assert store._artifacts == {}
    assert store._versions == {}
    assert store._idempotency == {}


def test_publish_advances_revision_and_preserves_history(
    publication_repo: ArtifactPublicationRepository,
    artifact_repo: WorkArtifactRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    initial = _seed_initial(publication_repo)
    published = publication_repo.publish_version(_publish_command())
    assert published.artifact.revision == initial.artifact.revision + 1
    assert published.artifact.current_version_id == _VERSION_2
    assert (
        artifact_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id=_ARTIFACT,
        )
        == published.artifact
    )
    listed = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert listed == (initial.version, published.version)


def test_publish_temporal_regression_rejected_without_state_change(
    publication_repo: ArtifactPublicationRepository,
    artifact_repo: WorkArtifactRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    initial = _seed_initial(publication_repo)
    t1 = initial.artifact.updated_at
    with pytest.raises(WorkArtifactTemporalConflict):
        publication_repo.publish_version(
            _publish_command(
                expected_revision=0,
                artifact_updated_at=t1 - timedelta(seconds=1),
                idempotency_key="temporal-regression",
            ),
        )
    unchanged = artifact_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert unchanged == initial.artifact
    assert unchanged is not None
    assert unchanged.revision == initial.artifact.revision
    assert unchanged.current_version_id == initial.artifact.current_version_id
    assert (
        version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=_VERSION_2,
        )
        is None
    )
    history = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert history == (initial.version,)
    store = publication_repo._store
    assert not any(
        key[3] == "temporal-regression"
        for key in store._idempotency
        if key[2] == "artifact.publish"
    )


def test_publish_temporal_equality_accepted(
    publication_repo: ArtifactPublicationRepository,
    artifact_repo: WorkArtifactRepository,
) -> None:
    initial = _seed_initial(publication_repo)
    published = publication_repo.publish_version(
        _publish_command(
            expected_revision=0,
            artifact_updated_at=initial.artifact.updated_at,
        ),
    )
    assert published.artifact.updated_at == initial.artifact.updated_at
    loaded = artifact_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert loaded == published.artifact


def test_publish_stale_revision_conflicts_without_new_version(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    _seed_initial(publication_repo)
    publication_repo.publish_version(_publish_command(expected_revision=0))
    with pytest.raises(WorkArtifactRevisionConflict):
        publication_repo.publish_version(
            _publish_command(
                work_artifact_version_id=_VERSION_3,
                expected_revision=0,
            ),
        )
    assert (
        version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=_VERSION_3,
        )
        is None
    )


def test_publish_duplicate_version_rejected(publication_repo: ArtifactPublicationRepository) -> None:
    _seed_initial(publication_repo)
    with pytest.raises(WorkArtifactVersionAlreadyExists):
        publication_repo.publish_version(
            _publish_command(work_artifact_version_id=_VERSION_1, expected_revision=0),
        )


def test_publish_wrong_work_item_rejected(publication_repo: ArtifactPublicationRepository) -> None:
    _seed_initial(publication_repo)
    with pytest.raises(WorkArtifactNotFound):
        publication_repo.publish_version(_publish_command(work_item_id="other-work-item"))


def test_publish_wrong_scope_not_found(publication_repo: ArtifactPublicationRepository) -> None:
    _seed_initial(publication_repo)
    with pytest.raises(WorkArtifactNotFound):
        publication_repo.publish_version(
            _publish_command(tenant_id=_TENANT_B, workspace_id=_WORKSPACE_B),
        )


def test_publish_idempotency_replay_and_conflict(publication_repo: ArtifactPublicationRepository) -> None:
    _seed_initial(publication_repo)
    command = _publish_command(idempotency_key="publish-idem")
    published = publication_repo.publish_version(command)
    replay = publication_repo.publish_version(command)
    assert replay == published
    with pytest.raises(ArtifactPublicationIdempotencyConflict):
        publication_repo.publish_version(
            _publish_command(
                work_artifact_version_id=_VERSION_3,
                idempotency_key="publish-idem",
            ),
        )


def test_publish_idempotency_replay_returns_original_after_later_publication(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    _seed_initial(publication_repo)
    first_publish = publication_repo.publish_version(
        _publish_command(idempotency_key="publish-idem"),
    )
    publication_repo.publish_version(
        _publish_command(
            work_artifact_version_id=_VERSION_3,
            expected_revision=1,
            published_at=_LATER_PUBLISHED + timedelta(minutes=10),
            artifact_updated_at=_LATER_PUBLISHED + timedelta(minutes=10),
        ),
    )
    replay = publication_repo.publish_version(
        _publish_command(idempotency_key="publish-idem"),
    )
    assert replay == first_publish
    assert replay.artifact.revision == 1
    assert replay.artifact.current_version_id == _VERSION_2
    listed = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert len(listed) == 3
    assert listed[-1].work_artifact_version_id == _VERSION_3


def test_create_and_publish_idempotency_namespaces_do_not_collide(
    publication_repo: ArtifactPublicationRepository,
) -> None:
    created = publication_repo.create_artifact_with_initial_version(
        _create_command(idempotency_key="shared-key"),
    )
    published = publication_repo.publish_version(
        _publish_command(
            expected_revision=0,
            idempotency_key="shared-key",
        ),
    )
    assert created.version.work_artifact_version_id == _VERSION_1
    assert published.version.work_artifact_version_id == _VERSION_2


def test_concurrent_publish_one_wins(
    publication_repo: ArtifactPublicationRepository,
    artifact_repo: WorkArtifactRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    initial = _seed_initial(publication_repo)
    errors: list[BaseException] = []
    results: list[PublishedWorkArtifactVersion] = []
    barrier = threading.Barrier(2)

    def attempt(version_id: str) -> None:
        try:
            barrier.wait(timeout=5)
            result = publication_repo.publish_version(
                _publish_command(
                    work_artifact_version_id=version_id,
                    expected_revision=0,
                ),
            )
            results.append(result)
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
    loser_version_id = (
        _VERSION_3
        if winner.version.work_artifact_version_id == _VERSION_2
        else _VERSION_2
    )

    final_artifact = artifact_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert final_artifact is not None
    assert final_artifact.revision == initial.artifact.revision + 1
    assert final_artifact.current_version_id == winner.version.work_artifact_version_id
    assert (
        version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=winner.version.work_artifact_version_id,
        )
        == winner.version
    )
    assert (
        version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=loser_version_id,
        )
        is None
    )
    history = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    assert history == (initial.version, winner.version)


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
        artifact_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            work_artifact_id=_ARTIFACT,
        )
        is None
    )
    assert (
        version_repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id=_VERSION_1,
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


def test_version_list_ordering_is_deterministic(
    publication_repo: ArtifactPublicationRepository,
    version_repo: WorkArtifactVersionRepository,
) -> None:
    _seed_initial(publication_repo)
    publication_repo.publish_version(_publish_command())
    publication_repo.publish_version(
        _publish_command(
            work_artifact_version_id=_VERSION_3,
            expected_revision=1,
            published_at=_LATER_PUBLISHED + timedelta(minutes=1),
            artifact_updated_at=_LATER_PUBLISHED + timedelta(minutes=1),
        ),
    )
    listed = version_repo.list_for_artifact(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_id=_ARTIFACT,
    )
    sort_keys = [(item.published_at, item.work_artifact_version_id) for item in listed]
    assert sort_keys == sorted(sort_keys)
