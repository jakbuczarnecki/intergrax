# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register SQLite in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_relational_store
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.relational_store.sqlite.contract_spec import CONTRACT_SPECS


def register_sqlite_integration(*, override: bool = False) -> None:
    register_from_manifest(SQLITE, create_sqlite_relational_store, override=override, contract_specs=CONTRACT_SPECS)
