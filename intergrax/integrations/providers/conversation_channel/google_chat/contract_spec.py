# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Google Chat conversation channel."""

from __future__ import annotations

from intergrax.integrations.providers.conversation_channel.google_chat.bundle import (
    create_google_chat_conversation_channel_integration,
)
from intergrax.integrations.providers.conversation_channel.google_chat.integration import (
    GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
    GoogleChatConversationChannelIntegration,
    GoogleChatConversationChannelIntegrationConfig,
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
    provider_id=GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
    integration_class=GoogleChatConversationChannelIntegration,
    contract_class=ConversationChannelIntegrationContract,
    contract_factory=create_google_chat_conversation_channel_integration,
    display_name="Google Chat",
    config_class=GoogleChatConversationChannelIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=False,
    supports_health_check=False,
    metadata={
        "source": "explicit_provider_declaration",
        "conversation_features": ("text", "single_choice"),
        "feature_declaration": "contract_intent",
        "runtime_implemented": False,
        "runtime_binding_supported": False
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
