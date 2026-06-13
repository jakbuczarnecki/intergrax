# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.health.category_probes import (
    HEALTH_CHECK_CODECRAFT_TOOL_ID,
    HEALTH_CHECK_GRAPH_STORE_TOOL_ID,
    HEALTH_CHECK_IDENTITY_PROVIDER_TOOL_ID,
    HEALTH_CHECK_KEY_VALUE_CACHE_TOOL_ID,
    HEALTH_CHECK_MESSAGE_BUS_TOOL_ID,
    HEALTH_CHECK_NOTIFICATION_CHANNEL_TOOL_ID,
    HEALTH_CHECK_OBJECT_STORAGE_TOOL_ID,
    HEALTH_CHECK_RELATIONAL_STORE_TOOL_ID,
    HEALTH_CHECK_SEARCH_PROVIDER_TOOL_ID,
    HEALTH_CHECK_WIKI_KNOWLEDGE_TOOL_ID,
)
from intergrax.tools.providers.health.contracts import (
    HealthCheckIntegrationInput,
    HealthCheckIntegrationOutput,
    HealthCheckProfileInput,
    HealthCheckProfileOutput,
)
from intergrax.tools.providers.health.handlers import (
    HealthCheckGraphStoreHandler,
    HealthCheckCodecraftHandler,
    HealthCheckIdentityProviderHandler,
    HealthCheckIntegrationHandler,
    HealthCheckKeyValueCacheHandler,
    HealthCheckMessageBusHandler,
    HealthCheckNotificationChannelHandler,
    HealthCheckObjectStorageHandler,
    HealthCheckProfileHandler,
    HealthCheckRelationalStoreHandler,
    HealthCheckSearchProviderHandler,
    HealthCheckWikiKnowledgeHandler,
)
from intergrax.tools.providers.health.service import (
    HEALTH_CHECK_INTEGRATION_TOOL_ID,
    HEALTH_CHECK_PROFILE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

HEALTH_BUNDLE_ID = "health"
HEALTH_TOOL_IDS: tuple[str, ...] = (
    HEALTH_CHECK_INTEGRATION_TOOL_ID,
    HEALTH_CHECK_PROFILE_TOOL_ID,
    HEALTH_CHECK_OBJECT_STORAGE_TOOL_ID,
    HEALTH_CHECK_KEY_VALUE_CACHE_TOOL_ID,
    HEALTH_CHECK_MESSAGE_BUS_TOOL_ID,
    HEALTH_CHECK_GRAPH_STORE_TOOL_ID,
    HEALTH_CHECK_IDENTITY_PROVIDER_TOOL_ID,
    HEALTH_CHECK_RELATIONAL_STORE_TOOL_ID,
    HEALTH_CHECK_WIKI_KNOWLEDGE_TOOL_ID,
    HEALTH_CHECK_SEARCH_PROVIDER_TOOL_ID,
    HEALTH_CHECK_NOTIFICATION_CHANNEL_TOOL_ID,
    HEALTH_CHECK_CODECRAFT_TOOL_ID,
)

_SLOT_PROBE_SPECS: tuple[tuple[str, str, type], ...] = (
    (
        HEALTH_CHECK_OBJECT_STORAGE_TOOL_ID,
        "Probe configured object storage backend health.",
        HealthCheckObjectStorageHandler,
    ),
    (
        HEALTH_CHECK_KEY_VALUE_CACHE_TOOL_ID,
        "Probe configured key-value cache backend health.",
        HealthCheckKeyValueCacheHandler,
    ),
    (
        HEALTH_CHECK_MESSAGE_BUS_TOOL_ID,
        "Probe configured message bus backend health.",
        HealthCheckMessageBusHandler,
    ),
    (
        HEALTH_CHECK_GRAPH_STORE_TOOL_ID,
        "Probe configured graph store backend health.",
        HealthCheckGraphStoreHandler,
    ),
    (
        HEALTH_CHECK_IDENTITY_PROVIDER_TOOL_ID,
        "Probe configured identity provider backend health.",
        HealthCheckIdentityProviderHandler,
    ),
    (
        HEALTH_CHECK_RELATIONAL_STORE_TOOL_ID,
        "Probe configured relational store backend health.",
        HealthCheckRelationalStoreHandler,
    ),
    (
        HEALTH_CHECK_WIKI_KNOWLEDGE_TOOL_ID,
        "Probe configured wiki knowledge backend health.",
        HealthCheckWikiKnowledgeHandler,
    ),
    (
        HEALTH_CHECK_SEARCH_PROVIDER_TOOL_ID,
        "Probe configured search provider backend health.",
        HealthCheckSearchProviderHandler,
    ),
    (
        HEALTH_CHECK_NOTIFICATION_CHANNEL_TOOL_ID,
        "Probe configured notification channel backend health.",
        HealthCheckNotificationChannelHandler,
    ),
    (
        HEALTH_CHECK_CODECRAFT_TOOL_ID,
        "Probe CodeCraft profile and sandbox substrate readiness.",
        HealthCheckCodecraftHandler,
    ),
)


def register_health_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=HEALTH_CHECK_INTEGRATION_TOOL_ID,
            name=HEALTH_CHECK_INTEGRATION_TOOL_ID,
            description="Run a health probe for a single integration catalog slug.",
            description_short="Probe integration slug.",
            input_schema=HealthCheckIntegrationInput,
            output_schema=HealthCheckIntegrationOutput,
            error_mapping={},
            side_effects=False,
            category="health",
            risk_level=ToolRiskLevel.LOW,
            tags=("health", "integration", "probe"),
        ),
        HealthCheckIntegrationHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HEALTH_CHECK_PROFILE_TOOL_ID,
            name=HEALTH_CHECK_PROFILE_TOOL_ID,
            description="Run health probes for all integrations configured in the host IntegrationProfile.",
            description_short="Probe integration profile.",
            input_schema=HealthCheckProfileInput,
            output_schema=HealthCheckProfileOutput,
            error_mapping={},
            side_effects=False,
            category="health",
            risk_level=ToolRiskLevel.LOW,
            tags=("health", "integration", "probe"),
        ),
        HealthCheckProfileHandler(ctx),
    )
    for tool_id, description, handler_cls in _SLOT_PROBE_SPECS:
        registry.register(
            ToolContract(
                tool_id=tool_id,
                name=tool_id,
                description=description,
                description_short=tool_id.split(".", 1)[-1],
                input_schema=HealthCheckProfileInput,
                output_schema=HealthCheckIntegrationOutput,
                error_mapping={},
                side_effects=False,
                category="health",
                risk_level=ToolRiskLevel.LOW,
                tags=("health", "integration", "probe", "slot"),
            ),
            handler_cls(ctx),
        )
