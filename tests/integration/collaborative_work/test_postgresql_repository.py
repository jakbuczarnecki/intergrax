# © Artur Czarnecki. All rights reserved.

"""Real PostgreSQL repository parity, concurrency, and production factory proofs."""

from __future__ import annotations

import threading

import pytest

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositories,
    open_postgresql_collaborative_work_repositories,
)
from intergrax.collaborative_work.repository import (
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import MembershipStatus, WorkspaceMembershipRole
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from tests.unit.collaborative_work import test_repository_contracts as contract_suite

pytestmark = [pytest.mark.integration, pytest.mark.network]


@pytest.fixture
def membership_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositories):
    return postgresql_collaborative_work_bundle.membership


@pytest.fixture
def delegation_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositories):
    return postgresql_collaborative_work_bundle.delegation


@pytest.fixture
def authority_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositories):
    return postgresql_collaborative_work_bundle.principal_authority


@pytest.fixture
def policy_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositories):
    return postgresql_collaborative_work_bundle.policy


@pytest.fixture
def profile_repo(postgresql_collaborative_work_bundle: CollaborativeWorkRepositories):
    return postgresql_collaborative_work_bundle.operation_profile


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


def test_postgresql_capabilities(postgresql_collaborative_work_bundle: CollaborativeWorkRepositories) -> None:
    caps = postgresql_collaborative_work_bundle.membership.capabilities
    assert caps.durable is True
    assert caps.reference_only is False
    assert caps.backend_id == "collaborative_work.postgresql"


def test_postgresql_concurrent_update_one_wins(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositories,
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
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositories,
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
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositories,
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
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositories,
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
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositories,
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
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositories,
) -> None:
    assert isinstance(postgresql_collaborative_work_bundle.store.schema_name, str)
    assert postgresql_collaborative_work_bundle.membership.capabilities.backend_id.endswith(
        "postgresql"
    )
