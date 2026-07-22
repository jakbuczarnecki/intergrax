# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for integration profile and cloud defaults."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import LOG, REDIS, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile

pytestmark = pytest.mark.unit


def test_default_lab_profile() -> None:
    profile = default_lab_profile()
    assert profile.relational_store is not None
    assert profile.relational_store.resolved_slug() == SQLITE.slug
    assert profile.notification_channel is not None
    assert profile.notification_channel.resolved_slug() == LOG.slug


def test_lab_classmethod() -> None:
    assert IntegrationProfile.lab() == default_lab_profile()


def test_profile_options_for_slug() -> None:
    profile = IntegrationProfile(
        key_value_cache=REDIS,
        options={REDIS: {"url": "redis://localhost"}},
    )
    assert profile.options_for_slug(REDIS) == {"url": "redis://localhost"}
    assert profile.options_for_slug("kafka") == {}


def test_azure_cloud_defaults() -> None:
    from intergrax.integrations.registry.catalog_manifests import AZURE

    profile = IntegrationProfile.with_cloud_platform(AZURE)
    assert profile.slug_for_category("object_storage") == "azure_blob"
    assert profile.slug_for_category("message_bus") == "service_bus"


def test_explicit_slug_overrides_cloud_default() -> None:
    from intergrax.integrations.registry.catalog_manifests import AWS
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations()
    profile = IntegrationProfile(
        cloud_platform=AWS,
        object_storage="filesystem",
    )
    assert profile.slug_for_category("object_storage") == "filesystem"


def test_rejects_wrong_slug_for_field() -> None:
    with pytest.raises(ValidationError):
        IntegrationProfile(key_value_cache=SQLITE)


def test_yaml_string_coercion() -> None:
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations()
    profile = IntegrationProfile.model_validate(
        {"relational_store": "sqlite", "key_value_cache": "redis"}
    )
    assert profile.relational_store is not None
    assert profile.relational_store.resolved_slug() == "sqlite"
    assert profile.key_value_cache is not None
    assert profile.key_value_cache.resolved_slug() == "redis"


def test_rejects_unknown_slug_string() -> None:
    clear_catalog()
    with pytest.raises(ValidationError):
        IntegrationProfile(relational_store="not_a_real_backend_xyz")


def test_profile_resolve_uses_typed_category(tmp_path) -> None:
    clear_catalog()
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(tmp_path)}},
    )
    store = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
    assert store is not None


def test_custom_manifest_without_enum(tmp_path) -> None:
    from intergrax.integrations.contracts.base import IntegrationStatus
    from intergrax.integrations.registry.plugin_register import register_from_manifest

    clear_catalog()

    custom = IntegrationManifest(
        slug="acme_warehouse",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        description="Custom warehouse plugin",
    )

    def _factory(**_kwargs: object) -> dict[str, str]:
        return {"backend": "acme"}

    register_from_manifest(custom, _factory)
    profile = IntegrationProfile(relational_store=custom)
    assert profile.relational_store is not None
    assert profile.relational_store.resolved_slug() == "acme_warehouse"
    resolved = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
    assert resolved == {"backend": "acme"}


def test_profile_accepts_prebuilt_instance() -> None:
    sentinel = object()
    profile = IntegrationProfile(relational_store=sentinel)
    assert profile.instance_for_category(IntegrationCategory.RELATIONAL_STORE) is sentinel


def test_integration_profile_json_roundtrip_preserves_slug_bindings() -> None:
    register_sqlite_integration()
    profile = IntegrationProfile(relational_store=SQLITE)
    restored = IntegrationProfile.model_validate(profile.model_dump(mode="json"))
    assert restored.slug_for_category(IntegrationCategory.RELATIONAL_STORE) == SQLITE.slug
