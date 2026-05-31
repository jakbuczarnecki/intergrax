# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Protocol conformance helpers for provider tests (Phase M.5)."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.object_storage import ObjectStorage


def assert_implements(instance: object, protocol: type[T]) -> T:
    if not isinstance(instance, protocol):
        raise AssertionError(
            f"Expected instance of {protocol.__name__}, got {type(instance)!r}"
        )
    return instance


def assert_relational_store(instance: object) -> RelationalStore:
    return assert_implements(instance, RelationalStore)


def assert_key_value_cache(instance: object) -> KeyValueCache:
    return assert_implements(instance, KeyValueCache)


def assert_message_bus(instance: object) -> MessageBus:
    return assert_implements(instance, MessageBus)


def assert_search_provider(instance: object) -> SearchProvider:
    return assert_implements(instance, SearchProvider)


def assert_notification_channel(instance: object) -> NotificationChannel:
    return assert_implements(instance, NotificationChannel)


def assert_interaction_surface(instance: object) -> InteractionSurface:
    return assert_implements(instance, InteractionSurface)


def assert_issue_tracker(instance: object) -> IssueTracker:
    return assert_implements(instance, IssueTracker)


def assert_wiki_knowledge(instance: object) -> WikiKnowledge:
    return assert_implements(instance, WikiKnowledge)


def assert_observability_backend(instance: object) -> ObservabilityBackend:
    return assert_implements(instance, ObservabilityBackend)


def assert_browser_automation(instance: object) -> BrowserAutomation:
    return assert_implements(instance, BrowserAutomation)


def assert_cloud_platform(instance: object) -> CloudPlatform:
    return assert_implements(instance, CloudPlatform)


def assert_collaboration_suite(instance: object) -> CollaborationSuite:
    return assert_implements(instance, CollaborationSuite)


def assert_document_store(instance: object) -> DocumentStore:
    return assert_implements(instance, DocumentStore)


def assert_object_storage(instance: object) -> ObjectStorage:
    return assert_implements(instance, ObjectStorage)


def assert_vector_store(instance: object) -> VectorStore:
    from intergrax.integrations.contracts.vector_store import VectorStore

    if not isinstance(instance, VectorStore):
        raise AssertionError(
            f"Expected instance of VectorStore, got {type(instance)!r}"
        )
    return instance
