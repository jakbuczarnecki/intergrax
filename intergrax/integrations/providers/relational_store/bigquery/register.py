# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register bigquery in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.bigquery.bundle import create_bigquery_relational_store
from intergrax.integrations.providers.relational_store.bigquery.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.relational_store.bigquery.contract_spec import CONTRACT_SPECS


def register_bigquery_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_bigquery_relational_store, override=override, contract_specs=CONTRACT_SPECS)
