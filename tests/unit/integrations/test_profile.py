# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for integration profile and cloud defaults."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit


def test_default_lab_profile() -> None:
    profile = default_lab_profile()
    assert profile.relational_store == IntegrationSlug.SQLITE
    assert profile.notification_channel == IntegrationSlug.LOG
    assert profile.interaction_surface == IntegrationSlug.LAB_JSON


def test_lab_classmethod() -> None:
    assert IntegrationProfile.lab() == default_lab_profile()


def test_profile_options_for_slug() -> None:
    profile = IntegrationProfile(
        key_value_cache=IntegrationSlug.REDIS,
        options={IntegrationSlug.REDIS: {"url": "redis://localhost"}},
    )
    assert profile.options_for_slug(IntegrationSlug.REDIS) == {"url": "redis://localhost"}
    assert profile.options_for_slug(IntegrationSlug.KAFKA) == {}


def test_azure_cloud_defaults() -> None:
    profile = IntegrationProfile.with_cloud_platform(IntegrationSlug.AZURE)
    assert profile.slug_for_category("object_storage") == IntegrationSlug.AZURE_BLOB.value
    assert profile.slug_for_category("message_bus") == IntegrationSlug.SERVICE_BUS.value


def test_explicit_slug_overrides_cloud_default() -> None:
    profile = IntegrationProfile(
        cloud_platform=IntegrationSlug.AWS,
        object_storage=IntegrationSlug.FILESYSTEM,
    )
    assert profile.slug_for_category("object_storage") == IntegrationSlug.FILESYSTEM.value


def test_rejects_wrong_slug_for_field() -> None:
    with pytest.raises(ValidationError):
        IntegrationProfile(key_value_cache=IntegrationSlug.SQLITE)


def test_yaml_string_coercion() -> None:
    profile = IntegrationProfile.model_validate(
        {"relational_store": "sqlite", "key_value_cache": "redis"}
    )
    assert profile.relational_store == IntegrationSlug.SQLITE
    assert profile.key_value_cache == IntegrationSlug.REDIS


def test_rejects_unknown_slug_string() -> None:
    with pytest.raises(ValidationError):
        IntegrationProfile(relational_store="not_a_real_backend")


def test_profile_resolve_uses_typed_category(tmp_path) -> None:
    from intergrax.integrations.contracts.base import IntegrationCategory
    from intergrax.integrations.providers.sqlite.register import register_sqlite_integration
    from intergrax.integrations.registry.catalog import clear_catalog

    clear_catalog()
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=IntegrationSlug.SQLITE,
        options={IntegrationSlug.SQLITE: {"data_dir": str(tmp_path)}},
    )
    store = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
    assert store.db_path == tmp_path / "intergrax.db"
