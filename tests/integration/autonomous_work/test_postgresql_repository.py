# © Artur Czarnecki. All rights reserved.

"""Real PostgreSQL Autonomous Work repository parity, concurrency, and recovery proofs."""

from __future__ import annotations

import multiprocessing
import threading
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from intergrax.autonomous_work.lifecycle import (
    WorkerLifecycleService,
    WorkerLifecycleTransitionRequest,
)
from intergrax.autonomous_work.persistence import (
    AutonomousWorkRepositories,
    open_postgresql_autonomous_work_repositories,
)
from intergrax.autonomous_work.postgresql_repository import (
    AutonomousWorkSchemaVersionError,
    PostgreSQLAutonomousWorkStore,
)
from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityConflict,
    AutonomousWorkRevisionConflict,
)
from intergrax.contracts.autonomous_work import (
    Revision,
    WorkerLifecycleState,
    initial_revision,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from tests.integration.autonomous_work.conftest import open_bundle, resolve_postgresql_config
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = [pytest.mark.integration, pytest.mark.network]

_UTC = timezone.utc


@pytest.fixture
def worker_definition_repo(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
):
    return postgresql_autonomous_work_bundle.worker_definition


@pytest.fixture
def worker_instance_repo(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
):
    return postgresql_autonomous_work_bundle.worker_instance


@pytest.fixture
def responsibility_repo(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
):
    return postgresql_autonomous_work_bundle.responsibility


@pytest.fixture
def worker_goal_repo(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
):
    return postgresql_autonomous_work_bundle.worker_goal


@pytest.fixture
def continuity_repo(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
):
    return postgresql_autonomous_work_bundle.work_continuity_state


def test_postgresql_worker_definition_contracts(worker_definition_repo: object) -> None:
    contract_suite.contract_worker_definition_create_get_and_version_history(
        worker_definition_repo
    )
    contract_suite.contract_worker_definition_idempotent_identical_create(
        worker_definition_repo
    )
    contract_suite.contract_worker_definition_same_revision_different_content_conflicts(
        worker_definition_repo
    )
    contract_suite.contract_worker_definition_missing_returns_none(worker_definition_repo)


def test_postgresql_worker_instance_contracts(worker_instance_repo: object) -> None:
    entity = contract_suite.worker_instance()
    contract_suite.contract_mutable_repository_create_and_get(
        worker_instance_repo,
        entity,
        load=lambda repo: repo.get(worker_instance_id=entity.worker_instance_id),
    )
    contract_suite.contract_mutable_repository_idempotent_identical_create(
        worker_instance_repo,
        entity,
    )
    contract_suite.contract_mutable_repository_same_id_different_content_conflicts(
        worker_instance_repo,
        entity,
        mutator=lambda value: replace(value, lifecycle_state=WorkerLifecycleState.ACTIVE),
    )
    contract_suite.contract_worker_instance_replace_advances_revision_and_does_not_mutate_input(
        worker_instance_repo
    )
    contract_suite.contract_worker_instance_stale_replace_conflicts_and_preserves_state(
        worker_instance_repo
    )
    contract_suite.contract_worker_instance_replace_missing_raises_not_found(worker_instance_repo)


def test_postgresql_responsibility_contracts(responsibility_repo: object) -> None:
    entity = contract_suite.responsibility()
    contract_suite.contract_mutable_repository_create_and_get(
        responsibility_repo,
        entity,
        load=lambda repo: repo.get(responsibility_id=entity.responsibility_id),
    )
    contract_suite.contract_mutable_repository_idempotent_identical_create(
        responsibility_repo,
        entity,
    )


def test_postgresql_worker_goal_contracts(worker_goal_repo: object) -> None:
    entity = contract_suite.worker_goal()
    contract_suite.contract_mutable_repository_create_and_get(
        worker_goal_repo,
        entity,
        load=lambda repo: repo.get(goal_id=entity.goal_id),
    )


def test_postgresql_continuity_contracts(continuity_repo: object) -> None:
    contract_suite.contract_continuity_repository_returns_latest_committed_state(continuity_repo)
    contract_suite.contract_continuity_state_worker_isolation(continuity_repo)
    contract_suite.contract_continuity_stale_revision_conflict(continuity_repo)


def test_postgresql_capabilities(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    caps = postgresql_autonomous_work_bundle.worker_instance.capabilities
    assert caps.durable is True
    assert caps.reference_only is False
    assert caps.backend_id == "autonomous_work.postgresql"


def test_postgresql_concurrent_update_one_wins(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    created = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance()
    )
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        bundle = open_postgresql_autonomous_work_repositories(
            config=postgresql_autonomous_work_bundle.store.config,
            schema_name=postgresql_autonomous_work_bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            bundle.worker_instance.replace(
                replace(created, lifecycle_state=WorkerLifecycleState.ACTIVE),
                expected_revision=created.revision,
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
    assert isinstance(errors[0], AutonomousWorkRevisionConflict)
    final = postgresql_autonomous_work_bundle.worker_instance.get(
        worker_instance_id=created.worker_instance_id
    )
    assert final is not None
    assert final.revision == Revision(created.revision.value + 1)


def _multiprocess_cas_attempt(
  dsn: str,
  schema_name: str,
  worker_instance_id: str,
  expected_revision: int,
  result_queue: multiprocessing.Queue[tuple[str, str | None]],
) -> None:
    config = PostgreSQLIntegrationConfig(dsn=dsn)
    bundle = open_postgresql_autonomous_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        loaded = bundle.worker_instance.get(worker_instance_id=worker_instance_id)
        assert loaded is not None
        bundle.worker_instance.replace(
            replace(loaded, lifecycle_state=WorkerLifecycleState.ACTIVE),
            expected_revision=Revision(expected_revision),
        )
        result_queue.put(("success", None))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__))
    finally:
        bundle.close()


def test_postgresql_multiprocess_cas_one_wins(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    config = resolve_postgresql_config()
    assert config is not None
    dsn = config.connection_string()
    assert dsn
    created = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance()
    )
    result_queue: multiprocessing.Queue[tuple[str, str | None]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_multiprocess_cas_attempt,
            args=(
                dsn,
                postgresql_autonomous_work_bundle.store.schema_name,
                created.worker_instance_id,
                created.revision.value,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    outcomes = [result_queue.get(timeout=10) for _ in processes]
    successes = [item for item in outcomes if item[0] == "success"]
    errors = [item for item in outcomes if item[0] == "error"]
    assert len(successes) == 1
    assert len(errors) == 1
    assert errors[0][1] == "AutonomousWorkRevisionConflict"
    final = postgresql_autonomous_work_bundle.worker_instance.get(
        worker_instance_id=created.worker_instance_id
    )
    assert final is not None
    assert final.revision == Revision(1)


def test_postgresql_idempotent_create_race(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    entity = contract_suite.worker_instance()
    results: list[object] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        bundle = open_postgresql_autonomous_work_repositories(
            config=postgresql_autonomous_work_bundle.store.config,
            schema_name=postgresql_autonomous_work_bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            results.append(bundle.worker_instance.create(entity))
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
    loaded = postgresql_autonomous_work_bundle.worker_instance.get(
        worker_instance_id=entity.worker_instance_id
    )
    assert loaded == results[0]


def test_postgresql_conflicting_create_race(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    base = contract_suite.worker_instance()
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt(mutator: object) -> None:
        bundle = open_postgresql_autonomous_work_repositories(
            config=postgresql_autonomous_work_bundle.store.config,
            schema_name=postgresql_autonomous_work_bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            bundle.worker_instance.create(mutator)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            bundle.close()

    threads = [
        threading.Thread(target=attempt, args=(base,)),
        threading.Thread(
            target=attempt,
            args=(replace(base, lifecycle_state=WorkerLifecycleState.ACTIVE),),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], AutonomousWorkEntityConflict)


def test_postgresql_restart_recovery(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    instance = contract_suite.worker_instance()
    continuity = contract_suite.continuity_state(
        worker_instance_ref=instance.worker_instance_id
    )
    config = postgresql_autonomous_work_bundle.store.config
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    postgresql_autonomous_work_bundle.worker_instance.create(instance)
    postgresql_autonomous_work_bundle.work_continuity_state.create(continuity)
    postgresql_autonomous_work_bundle.close()

    bundle_b = open_postgresql_autonomous_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        loaded_instance = bundle_b.worker_instance.get(
            worker_instance_id=instance.worker_instance_id
        )
        loaded_continuity = bundle_b.work_continuity_state.get(
            worker_instance_id=instance.worker_instance_id
        )
        assert loaded_instance == instance
        assert loaded_continuity == continuity

        service = WorkerLifecycleService(
            repository=bundle_b.worker_instance,
            clock=lambda: datetime(2026, 9, 2, 13, 0, tzinfo=_UTC),
        )
        result = service.transition(
            WorkerLifecycleTransitionRequest(
                worker_instance_id=instance.worker_instance_id,
                expected_revision=instance.revision,
                expected_state=WorkerLifecycleState.PROVISIONING,
                target_state=WorkerLifecycleState.ACTIVE,
                transition_reason="recovery-test",
            )
        )
        assert result.worker_instance.lifecycle_state == WorkerLifecycleState.ACTIVE
    finally:
        bundle_b.close()

    bundle_c = open_postgresql_autonomous_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        loaded = bundle_c.worker_instance.get(worker_instance_id=instance.worker_instance_id)
        assert loaded is not None
        assert loaded.lifecycle_state == WorkerLifecycleState.ACTIVE
        assert loaded.revision == Revision(1)
    finally:
        bundle_c.close()


def test_postgresql_fresh_and_repeated_bootstrap(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    config = postgresql_autonomous_work_bundle.store.config
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    entity = contract_suite.worker_instance()
    postgresql_autonomous_work_bundle.worker_instance.create(entity)

    bundle_reopen = open_postgresql_autonomous_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        loaded = bundle_reopen.worker_instance.get(
            worker_instance_id=entity.worker_instance_id
        )
        assert loaded == entity
    finally:
        bundle_reopen.close()


def test_postgresql_unsupported_schema_version_fails_closed(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    with postgresql_autonomous_work_bundle.store.transaction() as conn:
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (999,),
        )
    config = postgresql_autonomous_work_bundle.store.config
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    with pytest.raises(AutonomousWorkSchemaVersionError):
        open_postgresql_autonomous_work_repositories(
            config=config,
            schema_name=schema_name,
        )


def test_postgresql_transaction_rollback_preserves_state(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    created = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance()
    )
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    with store.transaction() as conn:
        conn.execute(
            """
            UPDATE aw_worker_instances
            SET record_json = %s, revision = %s
            WHERE worker_instance_id = %s
            """,
            ("{}", 99, created.worker_instance_id),
        )
        conn.rollback()
    loaded = postgresql_autonomous_work_bundle.worker_instance.get(
        worker_instance_id=created.worker_instance_id
    )
    assert loaded == created


def test_postgresql_unavailable_connection_fails_explicitly() -> None:
    config = PostgreSQLIntegrationConfig(
        dsn="postgresql://invalid:invalid@127.0.0.1:1/nonexistent",
    )
    with pytest.raises(IntegrationConfigurationError):
        open_postgresql_autonomous_work_repositories(config=config, schema_name="aw_fail_test")


def test_postgresql_multi_bundle_visibility(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    bundle_a = postgresql_autonomous_work_bundle
    bundle_b = open_postgresql_autonomous_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        created = bundle_a.worker_definition.create(contract_suite.worker_definition())
        loaded = bundle_b.worker_definition.get(
            worker_definition_id=created.worker_definition_id,
            definition_revision=created.revision,
        )
        assert loaded == created
    finally:
        bundle_b.close()


def test_postgresql_profile_reference_version_roundtrip(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    definition = contract_suite.worker_definition()
    created = postgresql_autonomous_work_bundle.worker_definition.create(definition)
    loaded = postgresql_autonomous_work_bundle.worker_definition.get(
        worker_definition_id=created.worker_definition_id,
        definition_revision=created.revision,
    )
    assert loaded == created
    assert loaded is not None
    assert loaded.governance_profile_ref.version == definition.governance_profile_ref.version


def test_postgresql_factory_has_no_in_memory_fallback(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    assert isinstance(postgresql_autonomous_work_bundle.store.schema_name, str)
    assert postgresql_autonomous_work_bundle.worker_instance.capabilities.backend_id.endswith(
        "postgresql"
    )
