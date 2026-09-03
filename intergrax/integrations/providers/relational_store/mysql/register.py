# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register mysql in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.mysql.bundle import create_mysql_relational_store
from intergrax.integrations.providers.relational_store.mysql.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.relational_store.mysql.contract_spec import CONTRACT_SPECS


def register_mysql_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_mysql_relational_store, override=override, contract_specs=CONTRACT_SPECS)
