# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Slack conversation channel."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.slack.bundle import (
    create_slack_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.messaging import (
    ConversationChannelIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="conversation_channel",
    provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    integration_class=SlackConversationChannelIntegration,
    contract_class=ConversationChannelIntegrationContract,
    contract_factory=create_slack_conversation_channel_integration,
    display_name="Slack",
    config_class=SlackConversationChannelIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={
        "source": "explicit_provider_declaration",
        "conversation_features": ("text", "single_choice"),
        "feature_declaration": "contract_intent",
        "runtime_implemented": True,
        "runtime_binding_supported": True,
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
