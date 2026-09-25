# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for SQLite integration provider (Phase M.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations._shared.conformance import assert_relational_store
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.adapter import (
    _SQLiteRelationalStore,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    create_sqlite_relational_store,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    RELATIONAL_DB_NAME,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SqliteRelationalStoreIntegration,
)
from intergrax.integrations.providers.relational_store.sqlite.register import (
    register_sqlite_integration,
)
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

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
