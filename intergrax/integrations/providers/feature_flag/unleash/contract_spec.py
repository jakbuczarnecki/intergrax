# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Unleash feature flag."""

from __future__ import annotations

from intergrax.integrations.providers.feature_flag.unleash.bundle import (
    create_unleash_feature_flag_integration,
)
from intergrax.integrations.providers.feature_flag.unleash.integration import (
    UNLEASH_FEATURE_FLAG_PROVIDER_ID,
    UnleashFeatureFlagIntegration,
    UnleashFeatureFlagIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="feature_flag",
    provider_id=UNLEASH_FEATURE_FLAG_PROVIDER_ID,
    integration_class=UnleashFeatureFlagIntegration,
    contract_class=FeatureFlagIntegrationContract,
    contract_factory=create_unleash_feature_flag_integration,
    display_name="Unleash",
    config_class=UnleashFeatureFlagIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
