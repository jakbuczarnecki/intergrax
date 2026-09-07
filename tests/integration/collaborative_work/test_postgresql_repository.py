# © Artur Czarnecki. All rights reserved.

"""Real PostgreSQL repository parity, concurrency, and production factory proofs."""

from __future__ import annotations

import os
import threading

import pytest

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositoriesWithSharedWork,
    open_postgresql_collaborative_work_repositories,
)
from intergrax.collaborative_work.postgresql_cross_process_cas_proof import (
    run_postgresql_work_item_cross_process_cas_proof,
)
from intergrax.collaborative_work.repository import (
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    UpdateWorkItemCommand,
    UpdateWorkspaceMembershipCommand,
    WorkItemRevisionConflict,
    WorkItemScopeKey,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import MembershipStatus, WorkItemState, WorkspaceMembership, WorkspaceMembershipRole
from intergrax.collaborative_work.serialization import workspace_membership_to_json
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from tests.unit.collaborative_work import test_repository_contracts as contract_suite
from tests.unit.collaborative_work import test_shared_work_in_memory_repository as shared_work_suite

pytestmark = [pytest.mark.integration, pytest.mark.network]


@pytest.fixture
def membership_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork):
    return postgresql_collaborative_work_bundle.membership


@pytest.fixture
def delegation_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork):
    return postgresql_collaborative_work_bundle.delegation


@pytest.fixture
def authority_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork):
    return postgresql_collaborative_work_bundle.principal_authority


@pytest.fixture
def policy_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork):
    return postgresql_collaborative_work_bundle.policy


@pytest.fixture
def profile_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork):
    return postgresql_collaborative_work_bundle.operation_profile


@pytest.fixture
def work_item_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork):
    return postgresql_collaborative_work_bundle.work_item


@pytest.fixture
def assignment_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork):
    return postgresql_collaborative_work_bundle.assignment


def test_postgresql_membership_create_get_revision_and_isolation(membership_repo: object) -> None:
    contract_suite.test_membership_create_get_revision_and_isolation(membership_repo)


def test_postgresql_membership_duplicate_and_stale_revision(membership_repo: object) -> None:
    contract_suite.test_membership_duplicate_and_stale_revision(membership_repo)


def test_postgresql_membership_idempotency_replay_after_update(membership_repo: object) -> None:
    contract_suite.test_membership_idempotency_replay_after_update(membership_repo)


def test_postgresql_delegation_create_update_idempotency(delegation_repo: object) -> None:
    contract_suite.test_delegation_create_update_idempotency(delegation_repo)


def test_postgresql_authority_grant_principal_uniqueness(authority_repo: object) -> None:
    contract_suite.test_authority_grant_principal_uniqueness(authority_repo)


def test_postgresql_policy_exact_key_uniqueness(policy_repo: object) -> None:
    contract_suite.test_policy_exact_key_uniqueness(policy_repo)


def test_postgresql_profile_revision_increment(profile_repo: object) -> None:
    contract_suite.test_profile_revision_increment(profile_repo)


def test_postgresql_capabilities(postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork) -> None:
    caps = postgresql_collaborative_work_bundle.membership.capabilities
    assert caps.durable is True
    assert caps.reference_only is False
    assert caps.backend_id == "collaborative_work.postgresql"
    work_caps = postgresql_collaborative_work_bundle.work_item.capabilities
    assert work_caps.durable is True
    assert work_caps.backend_id == "collaborative_work.postgresql"


def test_postgresql_concurrent_update_one_wins(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    created = postgresql_collaborative_work_bundle.membership.create(
        contract_suite._membership_command()
    )
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        bundle = open_postgresql_collaborative_work_repositories(
            config=postgresql_collaborative_work_bundle.store.config,
            schema_name=postgresql_collaborative_work_bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            bundle.membership.update(
                UpdateWorkspaceMembershipCommand(
                    scope=WorkspaceMembershipScopeKey(
                        tenant_id=contract_suite._TENANT_A,
                        workspace_id=contract_suite._WORKSPACE_A,
                        membership_id="membership-1",
                    ),
                    expected_revision=created.revision,
                    role=WorkspaceMembershipRole.ADMIN,
                    status=MembershipStatus.SUSPENDED,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            bundle.close()

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], WorkspaceMembershipRevisionConflict)
    final = postgresql_collaborative_work_bundle.membership.get(
        tenant_id=contract_suite._TENANT_A,
        workspace_id=contract_suite._WORKSPACE_A,
        membership_id="membership-1",
    )
    assert final is not None
    assert final.revision == created.revision + 1


def test_postgresql_unique_membership_create_race(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt(membership_id: str) -> None:
        bundle = open_postgresql_collaborative_work_repositories(
            config=postgresql_collaborative_work_bundle.store.config,
            schema_name=postgresql_collaborative_work_bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            bundle.membership.create(
                contract_suite._membership_command(membership_id=membership_id)
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            bundle.close()

    threads = [
        threading.Thread(target=attempt, args=("membership-a",)),
        threading.Thread(target=attempt, args=("membership-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], WorkspaceMembershipAlreadyExists)
    loaded = postgresql_collaborative_work_bundle.membership.get_for_principal(
        tenant_id=contract_suite._TENANT_A,
        workspace_id=contract_suite._WORKSPACE_A,
        principal_id="principal-1",
    )
    assert loaded is not None


def test_postgresql_idempotent_create_race(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    command = contract_suite._membership_command(idempotency_key="idem-race")
    results: list[object] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        bundle = open_postgresql_collaborative_work_repositories(
            config=postgresql_collaborative_work_bundle.store.config,
            schema_name=postgresql_collaborative_work_bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            results.append(bundle.membership.create(command))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            bundle.close()

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 2
    assert results[0] == results[1]
    loaded = postgresql_collaborative_work_bundle.membership.get_for_principal(
        tenant_id=contract_suite._TENANT_A,
        workspace_id=contract_suite._WORKSPACE_A,
        principal_id="principal-1",
    )
    assert loaded == results[0]


def test_postgresql_conflicting_idempotency_key(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    first = contract_suite._membership_command(
        membership_id="membership-1",
        principal_id="principal-1",
        idempotency_key="conflict-key",
    )
    postgresql_collaborative_work_bundle.membership.create(first)
    second = contract_suite._membership_command(
        membership_id="membership-2",
        principal_id="principal-2",
        idempotency_key="conflict-key",
    )
    with pytest.raises(WorkspaceMembershipIdempotencyConflict):
        postgresql_collaborative_work_bundle.membership.create(second)


def test_postgresql_multi_bundle_visibility(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    bundle_a = postgresql_collaborative_work_bundle
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        created = bundle_a.membership.create(contract_suite._membership_command())
        loaded = bundle_b.membership.get(
            tenant_id=contract_suite._TENANT_A,
            workspace_id=contract_suite._WORKSPACE_A,
            membership_id="membership-1",
        )
        assert loaded == created
    finally:
        bundle_b.close()


def test_postgresql_unavailable_connection_fails_explicitly() -> None:
    config = PostgreSQLIntegrationConfig(
        dsn="postgresql://invalid:invalid@127.0.0.1:1/nonexistent",
    )
    with pytest.raises(IntegrationConfigurationError):
        open_postgresql_collaborative_work_repositories(config=config, schema_name="cw_fail_test")


def test_postgresql_factory_has_no_sqlite_fallback(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    assert isinstance(postgresql_collaborative_work_bundle.store.schema_name, str)
    assert postgresql_collaborative_work_bundle.membership.capabilities.backend_id.endswith(
        "postgresql"
    )
    assert isinstance(postgresql_collaborative_work_bundle, CollaborativeWorkRepositoriesWithSharedWork)


def test_postgresql_work_item_create_and_get(work_item_repo: object) -> None:
    shared_work_suite.test_work_item_create_and_get(work_item_repo)  # type: ignore[arg-type]


def test_postgresql_work_item_duplicate_create_raises(work_item_repo: object) -> None:
    shared_work_suite.test_work_item_duplicate_create_raises(work_item_repo)  # type: ignore[arg-type]


def test_postgresql_work_item_scoped_isolation_read(work_item_repo: object) -> None:
    shared_work_suite.test_work_item_scoped_isolation_read(work_item_repo)  # type: ignore[arg-type]


def test_postgresql_work_item_update_success(work_item_repo: object) -> None:
    shared_work_suite.test_work_item_update_success(work_item_repo)  # type: ignore[arg-type]


def test_postgresql_work_item_stale_revision_conflict(work_item_repo: object) -> None:
    shared_work_suite.test_work_item_stale_revision_conflict_preserves_state(work_item_repo)  # type: ignore[arg-type]


def test_postgresql_work_item_idempotency_replay_and_conflict(work_item_repo: object) -> None:
    shared_work_suite.test_work_item_idempotency_replay_and_conflict(work_item_repo)  # type: ignore[arg-type]


def test_postgresql_work_item_idempotency_replay_after_update(work_item_repo: object) -> None:
    shared_work_suite.test_work_item_idempotency_replay_after_update_returns_original_create(
        work_item_repo,  # type: ignore[arg-type]
    )


def test_postgresql_work_item_concurrent_update_one_wins(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    """Cheaper same-process cross-connection CAS proof (not cross-process acceptance)."""
    bundle_a = postgresql_collaborative_work_bundle
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        created = bundle_a.work_item.create(shared_work_suite._create_work_item_command())
        read_a = bundle_a.work_item.get(
            tenant_id=shared_work_suite._TENANT_A,
            workspace_id=shared_work_suite._WORKSPACE_A,
            work_item_id="work-item-1",
        )
        read_b = bundle_b.work_item.get(
            tenant_id=shared_work_suite._TENANT_A,
            workspace_id=shared_work_suite._WORKSPACE_A,
            work_item_id="work-item-1",
        )
        assert read_a is not None and read_b is not None
        assert read_a.revision == read_b.revision == created.revision

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def attempt(bundle: CollaborativeWorkRepositoriesWithSharedWork) -> None:
            try:
                barrier.wait(timeout=5)
                bundle.work_item.update(
                    UpdateWorkItemCommand(
                        scope=WorkItemScopeKey(
                            tenant_id=shared_work_suite._TENANT_A,
                            workspace_id=shared_work_suite._WORKSPACE_A,
                            work_item_id="work-item-1",
                        ),
                        expected_revision=created.revision,
                        state=WorkItemState.ACTIVE,
                        updated_at=shared_work_suite._UPDATED_AT,
                    ),
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=attempt, args=(bundle_a,)),
            threading.Thread(target=attempt, args=(bundle_b,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 1
        assert isinstance(errors[0], WorkItemRevisionConflict)
        final = bundle_a.work_item.get(
            tenant_id=shared_work_suite._TENANT_A,
            workspace_id=shared_work_suite._WORKSPACE_A,
            work_item_id="work-item-1",
        )
        assert final is not None
        assert final.revision == created.revision + 1
    finally:
        bundle_b.close()


def test_postgresql_work_item_cross_process_cas_one_wins(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    bundle = postgresql_collaborative_work_bundle
    work_item_id = "work-item-cross-process"
    created = bundle.work_item.create(
        shared_work_suite._create_work_item_command(work_item_id=work_item_id),
    )
    parent_pid = os.getpid()
    result = run_postgresql_work_item_cross_process_cas_proof(
        config=bundle.store.config,
        schema_name=bundle.store.schema_name,
        tenant_id=shared_work_suite._TENANT_A,
        workspace_id=shared_work_suite._WORKSPACE_A,
        work_item_id=work_item_id,
        expected_revision=created.revision,
        updated_at=shared_work_suite._UPDATED_AT,
    )

    assert result.successes == 1
    assert result.conflicts == 1
    assert result.final_revision == created.revision + 1
    assert len(result.worker_pids) == 2
    assert len(set(result.worker_pids)) == 2
    assert all(pid != parent_pid for pid in result.worker_pids)


def test_postgresql_work_item_idempotency_survives_rematerialization(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    bundle_a = postgresql_collaborative_work_bundle
    command = shared_work_suite._create_work_item_command(idempotency_key="pg-idem-remat")
    created = bundle_a.work_item.create(command)
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        replayed = bundle_b.work_item.create(command)
        assert replayed == created
        assert replayed.revision == INITIAL_RECORD_REVISION
    finally:
        bundle_b.close()


def test_postgresql_assignment_create_and_get(assignment_repo: object) -> None:
    shared_work_suite.test_assignment_create_and_get(assignment_repo)  # type: ignore[arg-type]


def test_postgresql_assignment_duplicate_create_raises(assignment_repo: object) -> None:
    shared_work_suite.test_assignment_duplicate_create_raises(assignment_repo)  # type: ignore[arg-type]


def test_postgresql_assignment_multiple_for_same_work_item(assignment_repo: object) -> None:
    shared_work_suite.test_assignment_multiple_for_same_work_item_and_principals(
        assignment_repo,  # type: ignore[arg-type]
    )


def test_postgresql_assignment_update_preserves_identity(assignment_repo: object) -> None:
    shared_work_suite.test_assignment_update_preserves_identity(assignment_repo)  # type: ignore[arg-type]


def test_postgresql_assignment_idempotency_replay_and_conflict(assignment_repo: object) -> None:
    shared_work_suite.test_assignment_idempotency_replay_and_conflict(assignment_repo)  # type: ignore[arg-type]


def test_postgresql_mp2_schema_additive_preserves_mp1_data(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    bundle = postgresql_collaborative_work_bundle
    membership = WorkspaceMembership(
        membership_id="membership-legacy",
        tenant_id=contract_suite._TENANT_A,
        workspace_id=contract_suite._WORKSPACE_A,
        principal_id="principal-legacy",
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
        revision=0,
    )
    with bundle.store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO workspace_memberships (
                tenant_id, workspace_id, membership_id, principal_id, record_json, revision
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                membership.tenant_id,
                membership.workspace_id,
                membership.membership_id,
                membership.principal_id,
                workspace_membership_to_json(membership),
                membership.revision,
            ),
        )
        conn.execute(
            """
            INSERT INTO collaborative_idempotency (
                tenant_id, workspace_id, entity_kind, idempotency_key,
                semantic_fingerprint, result_json
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                contract_suite._TENANT_A,
                contract_suite._WORKSPACE_A,
                "workspace_membership",
                "legacy-idem",
                "legacy-fingerprint",
                workspace_membership_to_json(membership),
            ),
        )

    loaded = bundle.membership.get(
        tenant_id=contract_suite._TENANT_A,
        workspace_id=contract_suite._WORKSPACE_A,
        membership_id="membership-legacy",
    )
    assert loaded is not None
    assert loaded.principal_id == "principal-legacy"

    created = bundle.work_item.create(
        shared_work_suite._create_work_item_command(work_item_id="mp2-after-legacy"),
    )
    assert bundle.work_item.get(
        tenant_id=shared_work_suite._TENANT_A,
        workspace_id=shared_work_suite._WORKSPACE_A,
        work_item_id="mp2-after-legacy",
    ) == created
