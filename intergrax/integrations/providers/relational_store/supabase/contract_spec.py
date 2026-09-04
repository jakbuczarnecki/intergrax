# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Supabase relational store."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.supabase.bundle import (
    create_supabase_relational_store_integration,
)
from intergrax.integrations.providers.relational_store.supabase.integration import (
    SUPABASE_RELATIONAL_STORE_PROVIDER_ID,
    SupabaseRelationalStoreIntegration,
    SupabaseRelationalStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="relational_store",
    provider_id=SUPABASE_RELATIONAL_STORE_PROVIDER_ID,
    integration_class=SupabaseRelationalStoreIntegration,
    contract_class=RelationalStoreIntegrationContract,
    contract_factory=create_supabase_relational_store_integration,
    display_name="Supabase",
    config_class=SupabaseRelationalStoreIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
