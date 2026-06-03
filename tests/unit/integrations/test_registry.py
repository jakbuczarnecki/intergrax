# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os
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
from intergrax.integrations.registry.catalog_manifests import LAB_JSON, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def _register_fake(
    slug: str,
    *,
    categories: tuple[IntegrationCategory, ...],
    payload: str,
) -> None:
    register_integration(
        IntegrationEntry(
            slug=slug,
            categories=categories,
            factory=lambda **_: {"slug": slug, "payload": payload},
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

    assert instance == {"slug": "redis", "payload": "kv"}


def test_resolve_from_profile() -> None:
    _register_fake("sqlite", categories=(IntegrationCategory.RELATIONAL_STORE,), payload="sql")
    profile = IntegrationProfile(relational_store=SQLITE)

    instance = resolve(IntegrationCategory.RELATIONAL_STORE, profile=profile)

    assert instance["slug"] == "sqlite"


def test_resolve_with_typed_slug() -> None:
    _register_fake("redis", categories=(IntegrationCategory.KEY_VALUE_CACHE,), payload="kv")
    instance = resolve(IntegrationCategory.KEY_VALUE_CACHE, slug="redis")
    assert instance["slug"] == "redis"


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
    assert profile.interaction_surface is not None
    assert profile.interaction_surface.resolved_slug() == LAB_JSON.slug


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

    def _factory(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return captured

    register_integration(
        IntegrationEntry(
            slug="redis",
            categories=(IntegrationCategory.KEY_VALUE_CACHE,),
            factory=_factory,
        )
    )
    profile = IntegrationProfile(
        key_value_cache="redis",
        options={"redis": {"url": "redis://localhost/0"}},
    )

    resolve(IntegrationCategory.KEY_VALUE_CACHE, profile=profile)

    assert captured == {"url": "redis://localhost/0"}


def test_get_entry_metadata() -> None:
    register_integration(
        IntegrationEntry(
            slug="lab_json",
            categories=(IntegrationCategory.INTERACTION_SURFACE,),
            factory=lambda: object(),
            env_prefix="INTERGRAX_LAB",
        )
    )
    entry = get_entry("lab_json")
    assert entry.metadata.slug == "lab_json"
    assert entry.metadata.env_prefix == "INTERGRAX_LAB"
