# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform integration contracts (Tier-1 runtime)."""

from intergrax.runtime.integrations.contracts import (
    PLATFORM_INTEGRATION_CONTRACT_SCHEMA,
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationContract,
    PlatformIntegrationHealth,
    PlatformIntegrationKind,
    PlatformIntegrationSecurityPosture,
    PlatformIntegrationStatus,
    derive_platform_integration_id,
)

__all__ = [
    "PLATFORM_INTEGRATION_CONTRACT_SCHEMA",
    "PlatformIntegrationCapability",
    "PlatformIntegrationConfig",
    "PlatformIntegrationContract",
    "PlatformIntegrationHealth",
    "PlatformIntegrationKind",
    "PlatformIntegrationSecurityPosture",
    "PlatformIntegrationStatus",
    "derive_platform_integration_id",
]
