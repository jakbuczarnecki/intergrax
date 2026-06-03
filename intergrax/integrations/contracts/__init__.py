# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.contracts.base import (
    PROFILE_FIELD_BY_CATEGORY,
    HealthStatus,
    IntegrationCategory,
    IntegrationEntry,
    IntegrationDependencyError,
    IntegrationError,
    IntegrationFactory,
    IntegrationMetadata,
    IntegrationStatus,
    UnknownIntegrationCategoryError,
    UnknownIntegrationError,
    categories_for_profile_field,
    normalize_category,
)
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.collaboration_suite import (
    CalendarEvent,
    CalendarEventsResult,
    CollaborationSuite,
    MailListResult,
    MailMessage,
    UserRecord,
)
from intergrax.integrations.contracts.document_store import (
    DocumentQueryResult,
    DocumentRecord,
    DocumentStore,
)
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
from intergrax.integrations.contracts.object_storage import (
    ObjectStorage,
    PresignedUrlMethod,
    StoredObject,
)
from intergrax.integrations.contracts.notification_channel import (
    NotificationAdapter,
    NotificationChannel,
)
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
)
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.contracts.graph_store import GraphNodeRecord, GraphQueryResult, GraphStore

__all__ = [
    "PROFILE_FIELD_BY_CATEGORY",
    "BrowserAutomation",
    "PageContent",
    "CloudPlatform",
    "CalendarEvent",
    "CalendarEventsResult",
    "CollaborationSuite",
    "DocumentQueryResult",
    "DocumentRecord",
    "DocumentStore",
    "HealthStatus",
    "IntegrationCategory",
    "IntegrationEntry",
    "IntegrationDependencyError",
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
    "MailListResult",
    "MailMessage",
    "MetricPoint",
    "MetricQueryResult",
    "MetricSeries",
    "ObservabilityBackend",
    "WikiKnowledge",
    "WikiPageRecord",
    "WikiSearchResult",
    "KeyValueCache",
    "MessageBus",
    "ObjectStorage",
    "PresignedUrlMethod",
    "StoredObject",
    "NotificationAdapter",
    "NotificationChannel",
    "RelationalStore",
    "SearchProvider",
    "MetadataFilter",
    "VectorStore",
    "VectorStoreHit",
    "SecretsStore",
    "GraphStore",
    "GraphNodeRecord",
    "GraphQueryResult",
    "TaskHandle",
    "TaskQueue",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    "UserRecord",
    "UnknownIntegrationCategoryError",
    "UnknownIntegrationError",
    "categories_for_profile_field",
    "normalize_category",
]
