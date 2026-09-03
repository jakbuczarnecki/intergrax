# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register cloud_sql in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.cloud_sql.bundle import create_cloud_sql_relational_store
from intergrax.integrations.providers.relational_store.cloud_sql.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.relational_store.cloud_sql.contract_spec import CONTRACT_SPECS


def register_cloud_sql_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_cloud_sql_relational_store, override=override, contract_specs=CONTRACT_SPECS)
