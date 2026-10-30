# © Artur Czarnecki. All rights reserved.

"""EBH-5 Package C v2 — sqlite/duckdb slug switch @ resolve_from_profile (qualification only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.providers.relational_store.duckdb.register import register_duckdb_integration
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract


def run_package_c_v2_relational_sqlite_duckdb_dual_slug(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    clear_catalog()
    register_sqlite_integration()
    register_duckdb_integration()
    sqlite_dir = tmp_path / "sqlite"
    duck_dir = tmp_path / "duckdb"
    sqlite_dir.mkdir()
    duck_dir.mkdir()
    sqlite_profile = IntegrationProfile(
        relational_store="sqlite",
        options={"sqlite": {"data_dir": str(sqlite_dir)}},
    )
    duck_profile = IntegrationProfile(
        relational_store="duckdb",
        options={"duckdb": {"data_dir": str(duck_dir)}},
    )
    sqlite_store = resolve_from_profile(sqlite_profile, IntegrationCategory.RELATIONAL_STORE)
    duck_store = resolve_from_profile(duck_profile, IntegrationCategory.RELATIONAL_STORE)
    assert isinstance(sqlite_store, RelationalStoreIntegrationContract)
    assert isinstance(duck_store, RelationalStoreIntegrationContract)
    sqlite_store.connect()
    duck_store.connect()
    sqlite_store.close()
    duck_store.close()


def run_package_c_v2_relational_unregistered_fail_closed() -> None:
    clear_catalog()
    with pytest.raises((UnknownIntegrationError, Exception)):
        resolve_from_profile(IntegrationProfile(), IntegrationCategory.RELATIONAL_STORE)
