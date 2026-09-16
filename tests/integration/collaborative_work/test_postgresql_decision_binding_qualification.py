# © Artur Czarnecki. All rights reserved.

"""MP-4R4 — live PostgreSQL CollaborativeDecisionBinding repository qualification."""

from __future__ import annotations

import threading

import pytest

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositoriesWithArtifacts,
    open_postgresql_collaborative_work_repositories,
)
from intergrax.collaborative_work.repository import (
    CreateCollaborativeDecisionBindingCommand,
    CreateWorkItemCommand,
)
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBindingIdempotencyConflict,
    mint_collaborative_decision_binding_id,
)
from intergrax.contracts.collaborative_work import WorkItemState
from tests.unit.collaborative_work.test_decision_binding_service import (
    _TENANT,
    _TENANT_B,
    _WORKSPACE,
    _WORKSPACE_B,
    _WORK_ITEM_ID,
    _ACTING,
    _NOW,
    _identity,
    _proposal,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _seed_work_item(bundle: CollaborativeWorkRepositoriesWithArtifacts) -> None:
    bundle.work_item.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            created_by_principal_id=_ACTING,
            state=WorkItemState.OPEN,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )


def _create_command(
    *,
    idempotency_key: str,
    proposal: object | None = None,
    binding_id: str | None = None,
) -> CreateCollaborativeDecisionBindingCommand:
    identity = _identity()
    resolved_proposal = proposal or _proposal(identity)
    resolved_binding_id = binding_id or mint_collaborative_decision_binding_id(
        idempotency_key=idempotency_key,
    )
    return CreateCollaborativeDecisionBindingCommand(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        binding_id=resolved_binding_id,
        work_item_id=_WORK_ITEM_ID,
        work_artifact_version=None,
        decision_proposal=resolved_proposal,  # type: ignore[arg-type]
        created_by_principal_id=_ACTING,
        created_at=_NOW,
        idempotency_key=idempotency_key,
    )


@pytest.fixture
def decision_binding_repo(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
):
    _seed_work_item(postgresql_collaborative_work_bundle)
    return postgresql_collaborative_work_bundle.decision_binding


def test_postgresql_decision_binding_create_read_round_trip(
    decision_binding_repo: object,
) -> None:
    repo = decision_binding_repo
    identity = _identity()
    proposal = _proposal(identity)
    created = repo.create(_create_command(idempotency_key="pg-bind-1", proposal=proposal))  # type: ignore[attr-defined]
    loaded = repo.get(  # type: ignore[attr-defined]
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        binding_id=created.binding_id,
    )
    assert loaded == created
    assert loaded.decision_proposal == proposal


def test_postgresql_decision_binding_tenant_isolation(
    decision_binding_repo: object,
) -> None:
    repo = decision_binding_repo
    created = repo.create(_create_command(idempotency_key="pg-bind-tenant"))  # type: ignore[attr-defined]
    assert (
        repo.get(  # type: ignore[attr-defined]
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE,
            binding_id=created.binding_id,
        )
        is None
    )


def test_postgresql_decision_binding_workspace_isolation(
    decision_binding_repo: object,
) -> None:
    repo = decision_binding_repo
    repo.create(_create_command(idempotency_key="pg-bind-ws"))  # type: ignore[attr-defined]
    assert (
        repo.list_for_work_item(  # type: ignore[attr-defined]
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE_B,
            work_item_id=_WORK_ITEM_ID,
        )
        == ()
    )


def test_postgresql_decision_binding_idempotent_replay(
    decision_binding_repo: object,
) -> None:
    repo = decision_binding_repo
    command = _create_command(idempotency_key="pg-bind-idem")
    first = repo.create(command)  # type: ignore[attr-defined]
    second = repo.create(command)
    assert second.binding_id == first.binding_id
    assert second == first


def test_postgresql_decision_binding_idempotency_conflict(
    decision_binding_repo: object,
) -> None:
    repo = decision_binding_repo
    repo.create(_create_command(idempotency_key="pg-bind-conflict"))  # type: ignore[attr-defined]
    other = _proposal(_identity())
    with pytest.raises(CollaborativeDecisionBindingIdempotencyConflict):
        repo.create(  # type: ignore[attr-defined]
            _create_command(idempotency_key="pg-bind-conflict", proposal=other),
        )


def test_postgresql_decision_binding_semantic_dedup(
    decision_binding_repo: object,
) -> None:
    repo = decision_binding_repo
    identity = _identity()
    proposal = _proposal(identity)
    first = repo.create(  # type: ignore[attr-defined]
        _create_command(idempotency_key="pg-sem-1", proposal=proposal),
    )
    second = repo.create(
        _create_command(idempotency_key="pg-sem-2", proposal=proposal),
    )
    assert second.binding_id == first.binding_id


def test_postgresql_decision_binding_concurrent_semantic_duplicate_one_row(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle_a = postgresql_collaborative_work_bundle
    _seed_work_item(bundle_a)
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        identity = _identity()
        proposal = _proposal(identity)
        command_a = _create_command(
            idempotency_key="pg-race-a",
            proposal=proposal,
            binding_id=mint_collaborative_decision_binding_id(idempotency_key="pg-race-a"),
        )
        command_b = _create_command(
            idempotency_key="pg-race-b",
            proposal=proposal,
            binding_id=mint_collaborative_decision_binding_id(idempotency_key="pg-race-b"),
        )
        results: list[object] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def attempt(bundle: CollaborativeWorkRepositoriesWithArtifacts, command: object) -> None:
            try:
                barrier.wait(timeout=5)
                results.append(bundle.decision_binding.create(command))  # type: ignore[arg-type]
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=attempt, args=(bundle_a, command_a)),
            threading.Thread(target=attempt, args=(bundle_b, command_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(results) == 2
        binding_ids = {record.binding_id for record in results}  # type: ignore[attr-defined]
        assert len(binding_ids) == 1

        with bundle_a.store.transaction() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM collaborative_decision_bindings
                WHERE tenant_id = %s AND workspace_id = %s AND work_item_id = %s
                """,
                (_TENANT.strip(), _WORKSPACE.strip(), _WORK_ITEM_ID.strip()),
            ).fetchone()
        assert row is not None
        assert int(row["cnt"]) == 1
    finally:
        bundle_b.close()


def test_postgresql_decision_binding_concurrent_idempotency_conflict(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle_a = postgresql_collaborative_work_bundle
    _seed_work_item(bundle_a)
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        idem_key = "pg-idem-race"
        proposal_a = _proposal(_identity())
        proposal_b = _proposal(_identity())
        command_a = _create_command(idempotency_key=idem_key, proposal=proposal_a)
        command_b = _create_command(idempotency_key=idem_key, proposal=proposal_b)
        results: list[object] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def attempt(bundle: CollaborativeWorkRepositoriesWithArtifacts, command: object) -> None:
            try:
                barrier.wait(timeout=5)
                results.append(bundle.decision_binding.create(command))  # type: ignore[arg-type]
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=attempt, args=(bundle_a, command_a)),
            threading.Thread(target=attempt, args=(bundle_b, command_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], CollaborativeDecisionBindingIdempotencyConflict)

        with bundle_a.store.transaction() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM collaborative_decision_bindings
                WHERE tenant_id = %s AND workspace_id = %s
                """,
                (_TENANT.strip(), _WORKSPACE.strip()),
            ).fetchone()
        assert row is not None
        assert int(row["cnt"]) == 1
    finally:
        bundle_b.close()
