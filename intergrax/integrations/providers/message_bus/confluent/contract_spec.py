# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Confluent message bus."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.confluent.bundle import (
    create_confluent_message_bus_integration,
)
from intergrax.integrations.providers.message_bus.confluent.integration import (
    CONFLUENT_MESSAGE_BUS_PROVIDER_ID,
    ConfluentMessageBusIntegration,
    ConfluentMessageBusIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.messaging import (
    MessageBusIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="message_bus",
    provider_id=CONFLUENT_MESSAGE_BUS_PROVIDER_ID,
    integration_class=ConfluentMessageBusIntegration,
    contract_class=MessageBusIntegrationContract,
    contract_factory=create_confluent_message_bus_integration,
    display_name="Confluent",
    config_class=ConfluentMessageBusIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={
        "source": "explicit_provider_declaration"
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
