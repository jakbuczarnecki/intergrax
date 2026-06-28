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
from intergrax.runtime.integrations.observability import (
    OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA,
    OBSERVABILITY_VENDOR_PAYLOAD_SCHEMA,
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorKind,
    ObservabilityVendorMappingResult,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
    map_envelope_to_vendor_payload,
    require_policy_sanitized_envelope,
)

__all__ = [
    "OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA",
    "OBSERVABILITY_VENDOR_PAYLOAD_SCHEMA",
    "PLATFORM_INTEGRATION_CONTRACT_SCHEMA",
    "ObservabilityVendorIntegrationConfig",
    "ObservabilityVendorIntegrationContract",
    "ObservabilityVendorKind",
    "ObservabilityVendorMappingResult",
    "ObservabilityVendorPayload",
    "ObservabilityVendorSignal",
    "PlatformIntegrationCapability",
    "PlatformIntegrationConfig",
    "PlatformIntegrationContract",
    "PlatformIntegrationHealth",
    "PlatformIntegrationKind",
    "PlatformIntegrationSecurityPosture",
    "PlatformIntegrationStatus",
    "derive_platform_integration_id",
    "map_envelope_to_vendor_payload",
    "require_policy_sanitized_envelope",
]
