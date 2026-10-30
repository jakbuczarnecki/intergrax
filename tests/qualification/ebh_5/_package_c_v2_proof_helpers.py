# © Artur Czarnecki. All rights reserved.

"""EBH-5 Package C v2 — dual catalog path via resolve_from_profile."""

from __future__ import annotations

from pathlib import Path

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract


def run_package_c_v2_relational_dual_path(tmp_path: Path) -> None:
    clear_catalog()
    register_sqlite_integration()
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    profile_a = IntegrationProfile(
        relational_store="sqlite",
        options={"sqlite": {"data_dir": str(dir_a)}},
    )
    profile_b = IntegrationProfile(
        relational_store="sqlite",
        options={"sqlite": {"data_dir": str(dir_b)}},
    )
    store_a = resolve_from_profile(profile_a, IntegrationCategory.RELATIONAL_STORE)
    store_b = resolve_from_profile(profile_b, IntegrationCategory.RELATIONAL_STORE)
    assert isinstance(store_a, RelationalStoreIntegrationContract)
    assert isinstance(store_b, RelationalStoreIntegrationContract)
    store_a.connect()
    store_b.connect()
    store_a.close()
    store_b.close()
