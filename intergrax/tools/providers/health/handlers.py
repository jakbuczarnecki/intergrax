# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.health.contracts import (
    HealthCheckIntegrationInput,
    HealthCheckIntegrationOutput,
    HealthCheckProfileInput,
    HealthCheckProfileOutput,
)
from intergrax.tools.providers.health.service import health_check_integration, health_check_profile
from intergrax.tools.providers.health.category_probes import (
    health_check_codecraft,
    health_check_graph_store,
    health_check_identity_provider,
    health_check_key_value_cache,
    health_check_message_bus,
    health_check_notification_channel,
    health_check_object_storage,
    health_check_relational_store,
    health_check_search_provider,
    health_check_wiki_knowledge,
)


class HealthCheckIntegrationHandler(
    ServiceToolHandler[HealthCheckIntegrationInput, HealthCheckIntegrationOutput]
):
    _service = health_check_integration


class HealthCheckProfileHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckProfileOutput]):
    _service = health_check_profile


class HealthCheckObjectStorageHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_object_storage


class HealthCheckKeyValueCacheHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_key_value_cache


class HealthCheckMessageBusHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_message_bus


class HealthCheckGraphStoreHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_graph_store


class HealthCheckIdentityProviderHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_identity_provider


class HealthCheckRelationalStoreHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_relational_store


class HealthCheckWikiKnowledgeHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_wiki_knowledge


class HealthCheckSearchProviderHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_search_provider


class HealthCheckNotificationChannelHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_notification_channel


class HealthCheckCodecraftHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckIntegrationOutput]):
    _service = health_check_codecraft
