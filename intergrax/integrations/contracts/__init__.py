# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.contracts.base import (
    PROFILE_FIELD_BY_CATEGORY,
    HealthStatus,
    IntegrationCategory,
    IntegrationEntry,
    IntegrationError,
    IntegrationFactory,
    IntegrationMetadata,
    IntegrationStatus,
    UnknownIntegrationCategoryError,
    UnknownIntegrationError,
    categories_for_profile_field,
    normalize_category,
)
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.interaction_surface import (
    InteractionAdapter,
    InteractionSurface,
)
from intergrax.integrations.contracts.issue_tracker import (
    IssueComment,
    IssueRecord,
    IssueSearchResult,
    IssueTracker,
)
from intergrax.integrations.contracts.observability_backend import (
    MetricPoint,
    MetricQueryResult,
    MetricSeries,
    ObservabilityBackend,
)
from intergrax.integrations.contracts.wiki_knowledge import (
    WikiKnowledge,
    WikiPageRecord,
    WikiSearchResult,
)
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import (
    MessageBus,
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from intergrax.integrations.contracts.notification_channel import (
    NotificationAdapter,
    NotificationChannel,
)
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider

__all__ = [
    "PROFILE_FIELD_BY_CATEGORY",
    "CloudPlatform",
    "HealthStatus",
    "IntegrationCategory",
    "IntegrationEntry",
    "IntegrationError",
    "IntegrationFactory",
    "IntegrationMetadata",
    "IntegrationStatus",
    "InteractionAdapter",
    "InteractionSurface",
    "IssueComment",
    "IssueRecord",
    "IssueSearchResult",
    "IssueTracker",
    "MetricPoint",
    "MetricQueryResult",
    "MetricSeries",
    "ObservabilityBackend",
    "WikiKnowledge",
    "WikiPageRecord",
    "WikiSearchResult",
    "KeyValueCache",
    "MessageBus",
    "NotificationAdapter",
    "NotificationChannel",
    "RelationalStore",
    "SearchProvider",
    "TaskHandle",
    "TaskQueue",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    "UnknownIntegrationCategoryError",
    "UnknownIntegrationError",
    "categories_for_profile_field",
    "normalize_category",
]
