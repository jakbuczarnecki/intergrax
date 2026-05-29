# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Protocol conformance helpers for provider tests (Phase M.5)."""

from __future__ import annotations

from typing import TypeVar

from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider

T = TypeVar("T")


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


def assert_cloud_platform(instance: object) -> CloudPlatform:
    return assert_implements(instance, CloudPlatform)
