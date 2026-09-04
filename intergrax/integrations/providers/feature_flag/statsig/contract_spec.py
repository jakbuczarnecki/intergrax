# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Statsig feature flag."""

from __future__ import annotations

from intergrax.integrations.providers.feature_flag.statsig.bundle import (
    create_statsig_feature_flag_integration,
)
from intergrax.integrations.providers.feature_flag.statsig.integration import (
    STATSIG_FEATURE_FLAG_PROVIDER_ID,
    StatsigFeatureFlagIntegration,
    StatsigFeatureFlagIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="feature_flag",
    provider_id=STATSIG_FEATURE_FLAG_PROVIDER_ID,
    integration_class=StatsigFeatureFlagIntegration,
    contract_class=FeatureFlagIntegrationContract,
    contract_factory=create_statsig_feature_flag_integration,
    display_name="Statsig",
    config_class=StatsigFeatureFlagIntegrationConfig,
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
