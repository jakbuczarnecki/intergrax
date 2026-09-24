# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite integration plugin — type-based registration alternative to manifest-only."""

from __future__ import annotations

from intergrax.integrations.contracts.catalog_factory import IntegrationFactoryConfigValue
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_relational_store
from intergrax.integrations.providers.relational_store.sqlite.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.relational_store.sqlite.integration import SqliteRelationalStoreIntegration
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.contract_spec import IntegrationContractSpec


class SqliteIntegrationPlugin:
    """Register via :func:`register_integration_plugin` or ``IntegrationProfile(relational_store=SqliteIntegrationPlugin)``."""

    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return SQLITE

    @classmethod
    def integration_contract_specs(cls) -> tuple[IntegrationContractSpec, ...]:
        return CONTRACT_SPECS

    @classmethod
    def create_integration(cls, **kwargs: IntegrationFactoryConfigValue) -> SqliteRelationalStoreIntegration:
        return create_sqlite_relational_store(**kwargs)
