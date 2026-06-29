# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for SQLite integration provider (Phase M.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.experiments.store import SQLiteExperimentStore
from intergrax.integrations._shared.conformance import assert_relational_store
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.adapter import _SQLiteRelationalStore
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    SQLiteIntegrationBundle,
    create_sqlite_integration,
    create_sqlite_relational_store,
    create_sqlite_trace_store,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    EXPERIMENTS_DB_NAME,
    RELATIONAL_DB_NAME,
    TRACE_DB_NAME,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import SqliteRelationalStoreIntegration
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
    SQLiteOrganizationProfileStore,
)
from intergrax.runtime.task_memory.stores.sqlite_task_memory_store import SQLiteTaskMemoryStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_sqlite_relational_store_execute_and_fetch(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = _SQLiteRelationalStore(db_path)
    assert_relational_store(store)

    store.connect()
    store.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    store.execute("INSERT INTO items (name) VALUES (?)", ("alpha",))
    rows = store.fetch_all("SELECT name FROM items")
    store.close()

    assert [row["name"] for row in rows] == ["alpha"]


def test_create_sqlite_integration_bundle_uses_shared_data_dir(tmp_path: Path) -> None:
    bundle = create_sqlite_integration(data_dir=tmp_path)

    assert isinstance(bundle, SQLiteIntegrationBundle)
    assert bundle.paths.data_dir == tmp_path
    assert bundle.paths.relational == tmp_path / RELATIONAL_DB_NAME
    assert bundle.paths.trace == tmp_path / TRACE_DB_NAME
    assert bundle.paths.experiments == tmp_path / EXPERIMENTS_DB_NAME

    assert isinstance(bundle.relational_store, SqliteRelationalStoreIntegration)
    assert isinstance(bundle.trace_store, SQLiteRunTraceStore)
    assert isinstance(bundle.runtime_event_store, SQLiteRuntimeEventStore)
    assert type(bundle.task_checkpoint_store).__name__ == "SQLiteTaskCheckpointStore"
    assert isinstance(bundle.human_decision_store, SQLiteHumanDecisionStore)
    assert isinstance(bundle.task_memory_store, SQLiteTaskMemoryStore)
    assert isinstance(bundle.experiment_store, SQLiteExperimentStore)
    assert isinstance(bundle.idempotency_store, SQLiteIdempotencyStore)
    assert isinstance(bundle.session_storage, SQLiteSessionStorage)
    assert isinstance(bundle.organization_profile_store, SQLiteOrganizationProfileStore)
    assert isinstance(bundle.user_profile_store, SQLiteUserProfileStore)

    assert bundle.relational_store.db_path.exists()
    assert bundle.paths.trace.exists()


def test_create_sqlite_trace_store_factory(tmp_path: Path) -> None:
    store = create_sqlite_trace_store(data_dir=tmp_path)
    assert isinstance(store, SQLiteRunTraceStore)


def test_register_and_resolve_via_lab_profile(tmp_path: Path) -> None:
    register_sqlite_integration()
    profile = IntegrationProfile(relational_store="sqlite")

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"data_dir": str(tmp_path)},
    )

    assert_relational_store(store)
    assert isinstance(store, SqliteRelationalStoreIntegration)
    assert store.db_path == tmp_path / RELATIONAL_DB_NAME


def test_register_default_integrations_includes_sqlite(tmp_path: Path) -> None:
    register_default_integrations()
    profile = IntegrationProfile.lab()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"data_dir": str(tmp_path)},
    )

    assert isinstance(store, SqliteRelationalStoreIntegration)


def test_create_sqlite_relational_store_catalog_factory(tmp_path: Path) -> None:
    store = create_sqlite_relational_store(data_dir=tmp_path)
    assert_relational_store(store)
    assert store.db_path == tmp_path / RELATIONAL_DB_NAME
