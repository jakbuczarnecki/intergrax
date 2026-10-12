# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, Mapping, Sequence

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationCategoryMismatchError,
    IntegrationConfigurationError,
    IntegrationEntry,
    IntegrationStatus,
    UnknownIntegrationError,
)
from intergrax.integrations.registry.catalog import (
    clear_catalog,
    get_entry,
    list_slugs,
    register_integration,
)
from intergrax.integrations.registry.factory import (
    build_profile_from_env,
    build_profile_from_mapping,
    resolve,
    resolve_slug,
)
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile
from intergrax.runtime.integrations.categories import (
    KeyValueCacheIntegrationContract,
    MessageBusIntegrationContract,
    NotificationChannelIntegrationContract,
    RelationalStoreIntegrationContract,
)

pytestmark = pytest.mark.unit


class _FakeKeyValueCacheIntegration(KeyValueCacheIntegrationContract):
    test_payload: str = ""


class _FakeRelationalStoreIntegration(RelationalStoreIntegrationContract):
    test_payload: str = ""


class _FakeMessageBusIntegration(MessageBusIntegrationContract):
    test_payload: str = ""


class _FakeNotificationChannelIntegration(NotificationChannelIntegrationContract):
    test_payload: str = ""


@pytest.fixture(autouse=True)
def _isolated_catalog() -> Iterator[None]:
    clear_catalog()
    yield
    clear_catalog()


def _materialize_fake(
    slug: str,
    payload: str,
    category: IntegrationCategory,
    **kwargs: Any,
) -> KeyValueCacheIntegrationContract | RelationalStoreIntegrationContract | MessageBusIntegrationContract:
    _ = kwargs
    if category is IntegrationCategory.KEY_VALUE_CACHE:
        return _FakeKeyValueCacheIntegration.for_provider(
            provider_id=slug,
            display_name=slug,
        ).model_copy(update={"test_payload": payload})
    if category is IntegrationCategory.RELATIONAL_STORE:
        return _FakeRelationalStoreIntegration.for_provider(
            provider_id=slug,
            display_name=slug,
        ).model_copy(update={"test_payload": payload})
    if category is IntegrationCategory.MESSAGE_BUS:
        return _FakeMessageBusIntegration.for_provider(
            provider_id=slug,
            display_name=slug,
        ).model_copy(update={"test_payload": payload})
    msg = f"unsupported fake category {category!r}"
    raise ValueError(msg)


def _register_fake(
    slug: str,
    *,
    categories: tuple[IntegrationCategory, ...],
    payload: str,
) -> None:
    primary = categories[0]

    def _factory(**kwargs: Any) -> KeyValueCacheIntegrationContract | RelationalStoreIntegrationContract | MessageBusIntegrationContract:
        category = kwargs.pop("integration_category", primary)
        return _materialize_fake(slug, payload, category, **kwargs)

    register_integration(
        IntegrationEntry(
            slug=slug,
            categories=categories,
            factory=_factory,
            status=IntegrationStatus.BETA,
        )
    )


def test_register_and_list_by_category() -> None:
    _register_fake("redis", categories=(IntegrationCategory.KEY_VALUE_CACHE,), payload="kv")
    _register_fake("sqlite", categories=(IntegrationCategory.RELATIONAL_STORE,), payload="sql")

    assert list_slugs() == ["redis", "sqlite"]
    assert list_slugs(category=IntegrationCategory.KEY_VALUE_CACHE) == ["redis"]


def test_resolve_with_explicit_slug() -> None:
    _register_fake("redis", categories=(IntegrationCategory.KEY_VALUE_CACHE,), payload="kv")

    instance = resolve(IntegrationCategory.KEY_VALUE_CACHE, slug="redis")

    assert instance.provider_id == "redis"
    assert getattr(instance, "test_payload") == "kv"


def test_resolve_from_profile() -> None:
    _register_fake("sqlite", categories=(IntegrationCategory.RELATIONAL_STORE,), payload="sql")
    profile = IntegrationProfile(relational_store=IntegrationBinding.from_slug("sqlite"))

    instance = resolve(IntegrationCategory.RELATIONAL_STORE, profile=profile)

    assert instance.provider_id == "sqlite"
    assert getattr(instance, "test_payload") == "sql"


def test_resolve_with_typed_slug() -> None:
    _register_fake("redis", categories=(IntegrationCategory.KEY_VALUE_CACHE,), payload="kv")
    instance = resolve(IntegrationCategory.KEY_VALUE_CACHE, slug="redis")
    assert instance.provider_id == "redis"


def test_resolve_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_fake("celery", categories=(IntegrationCategory.MESSAGE_BUS,), payload="bus")
    monkeypatch.setenv("INTERGRAX_INTEGRATION_MESSAGE_BUS", "celery")

    slug = resolve_slug(IntegrationCategory.MESSAGE_BUS)
    assert slug == "celery"


def test_build_profile_from_mapping() -> None:
    _register_fake("sqlite", categories=(IntegrationCategory.RELATIONAL_STORE,), payload="sql")
    _register_fake("redis", categories=(IntegrationCategory.KEY_VALUE_CACHE,), payload="kv")
    profile = build_profile_from_mapping(
        {
            "integrations": {
                "relational_store": "sqlite",
                "key_value_cache": "redis",
            }
        }
    )
    assert profile.relational_store is not None
    assert profile.relational_store.resolved_slug() == "sqlite"
    assert profile.key_value_cache is not None
    assert profile.key_value_cache.resolved_slug() == "redis"


def test_build_profile_from_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_fake("postgresql", categories=(IntegrationCategory.RELATIONAL_STORE,), payload="pg")
    monkeypatch.setenv("INTERGRAX_INTEGRATION_RELATIONAL_STORE", "postgresql")
    profile = build_profile_from_env(defaults=default_lab_profile())

    assert profile.relational_store is not None
    assert profile.relational_store.resolved_slug() == "postgresql"
    assert profile.notification_channel is not None
    assert profile.notification_channel.resolved_slug() is not None


def test_cloud_platform_defaults_when_category_unset() -> None:
    from intergrax.integrations.registry.catalog_manifests import AWS

    profile = IntegrationProfile.with_cloud_platform(AWS)

    assert profile.slug_for_category(IntegrationCategory.OBJECT_STORAGE) == "s3"
    assert profile.slug_for_category(IntegrationCategory.MESSAGE_BUS) == "sqs"
    assert profile.slug_for_category(IntegrationCategory.RELATIONAL_STORE) is None


def test_resolve_rejects_unregistered_slug() -> None:
    with pytest.raises((UnknownIntegrationError, ValueError), match="redis"):
        resolve(IntegrationCategory.KEY_VALUE_CACHE, slug="redis")


def test_resolve_rejects_unknown_slug() -> None:
    with pytest.raises(ValueError, match="Unknown integration slug"):
        resolve(IntegrationCategory.KEY_VALUE_CACHE, slug="missing")


def test_resolve_rejects_category_mismatch() -> None:
    _register_fake("redis", categories=(IntegrationCategory.KEY_VALUE_CACHE,), payload="kv")

    with pytest.raises(IntegrationCategoryMismatchError):
        resolve(IntegrationCategory.MESSAGE_BUS, slug="redis")


def test_resolve_requires_configuration() -> None:
    with pytest.raises(IntegrationConfigurationError):
        resolve(IntegrationCategory.RELATIONAL_STORE)


def test_factory_receives_profile_options() -> None:
    captured: dict[str, Any] = {}

    def _factory(**kwargs: Any) -> KeyValueCacheIntegrationContract:
        captured.update(kwargs)
        return _FakeKeyValueCacheIntegration.for_provider(
            provider_id="redis",
            display_name="redis",
        )

    register_integration(
        IntegrationEntry(
            slug="redis",
            categories=(IntegrationCategory.KEY_VALUE_CACHE,),
            factory=_factory,
        )
    )
    profile = IntegrationProfile(
        key_value_cache=IntegrationBinding.from_slug("redis"),
        options={"redis": {"url": "redis://localhost/0"}},
    )

    resolve(IntegrationCategory.KEY_VALUE_CACHE, profile=profile)

    assert captured == {"url": "redis://localhost/0"}


def test_get_entry_metadata() -> None:
    register_integration(
        IntegrationEntry(
            slug="lab_json",
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=lambda: _FakeNotificationChannelIntegration.for_provider(
                provider_id="lab_json",
                display_name="lab_json",
            ),
            env_prefix="INTERGRAX_LAB",
        )
    )
    entry = get_entry("lab_json")
    assert entry.metadata.slug == "lab_json"
    assert entry.metadata.env_prefix == "INTERGRAX_LAB"
