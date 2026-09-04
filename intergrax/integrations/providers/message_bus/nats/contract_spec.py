# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Nats message bus."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.nats.bundle import (
    create_nats_message_bus_integration,
)
from intergrax.integrations.providers.message_bus.nats.integration import (
    NATS_MESSAGE_BUS_PROVIDER_ID,
    NatsMessageBusIntegration,
    NatsMessageBusIntegrationConfig,
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
    provider_id=NATS_MESSAGE_BUS_PROVIDER_ID,
    integration_class=NatsMessageBusIntegration,
    contract_class=MessageBusIntegrationContract,
    contract_factory=create_nats_message_bus_integration,
    display_name="Nats",
    config_class=NatsMessageBusIntegrationConfig,
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
