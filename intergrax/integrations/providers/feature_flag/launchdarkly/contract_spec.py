# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Launchdarkly feature flag."""

from __future__ import annotations

from intergrax.integrations.providers.feature_flag.launchdarkly.bundle import (
    create_launchdarkly_feature_flag_integration,
)
from intergrax.integrations.providers.feature_flag.launchdarkly.integration import (
    LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID,
    LaunchdarklyFeatureFlagIntegration,
    LaunchdarklyFeatureFlagIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="feature_flag",
    provider_id=LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID,
    integration_class=LaunchdarklyFeatureFlagIntegration,
    contract_class=FeatureFlagIntegrationContract,
    contract_factory=create_launchdarkly_feature_flag_integration,
    display_name="Launchdarkly",
    config_class=LaunchdarklyFeatureFlagIntegrationConfig,
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
