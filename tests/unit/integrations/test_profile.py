# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for integration profile and cloud defaults."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import LOG, REDIS, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.core.ref import validate_integration_ref
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.contract_metadata import IntegrationContractMetadataError
from intergrax.runtime.integrations.contracts import PlatformIntegrationContract, PlatformIntegrationSecurityPosture
from intergrax.runtime.integrations.contracts import PlatformIntegrationCapability

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
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations()
    profile = IntegrationProfile(key_value_cache=SQLITE)
    with pytest.raises(ValueError, match="not valid for profile field"):
        validate_integration_ref("key_value_cache", profile.key_value_cache)


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
    with pytest.raises(ValueError, match="Unknown integration slug"):
        validate_integration_ref("relational_store", "not_a_real_backend_xyz")


def test_profile_resolve_uses_typed_category(tmp_path) -> None:
    clear_catalog()
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(tmp_path)}},
    )
    store = resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)
    assert store is not None
    assert isinstance(store, RelationalStoreIntegrationContract)


class _SampleRelationalIntegration(RelationalStoreIntegrationContract):
    pass


def test_custom_manifest_with_category_contract(tmp_path) -> None:
    clear_catalog()

    custom = IntegrationManifest(
        slug="acme_warehouse",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        description="Custom warehouse plugin",
    )

    def _factory(**_kwargs: object) -> _SampleRelationalIntegration:
        return _SampleRelationalIntegration.for_provider(
            provider_id="acme_warehouse",
            display_name="Acme warehouse",
            config=CategoryIntegrationConfig(enabled=True),
        )

    spec = declare_integration_contract(
        category="relational_store",
        provider_id="acme_warehouse",
        integration_class=_SampleRelationalIntegration,
        contract_factory=_factory,
        display_name="Acme warehouse",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
    )
    register_from_manifest(custom, _factory, contract_specs=(spec,))
    profile = IntegrationProfile(relational_store=custom)
    assert profile.relational_store is not None
    assert profile.relational_store.resolved_slug() == "acme_warehouse"
    resolved = resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)
    assert isinstance(resolved, RelationalStoreIntegrationContract)
    assert resolved.provider_id == "acme_warehouse"


def test_prebuilt_relational_store_requires_category_contract() -> None:
    integration = _SampleRelationalIntegration.for_provider(
        provider_id="injected_sql",
        display_name="Injected SQL",
    )
    profile = IntegrationProfile(relational_store=integration)
    resolved = profile.instance_for_category(IntegrationCategory.RELATIONAL_STORE)
    assert resolved is integration


def test_prebuilt_plain_object_fail_closed() -> None:
    profile = IntegrationProfile(relational_store=object())
    with pytest.raises(TypeError, match="expected RelationalStoreIntegrationContract"):
        profile.instance_for_category(IntegrationCategory.RELATIONAL_STORE)


def test_prebuilt_wrong_category_contract_fail_closed() -> None:
    wrong = VectorStoreIntegrationContract.for_provider(
        provider_id="wrong_vector",
        display_name="Wrong vector",
    )
    profile = IntegrationProfile(relational_store=wrong)
    with pytest.raises(TypeError, match="expected RelationalStoreIntegrationContract"):
        profile.instance_for_category(IntegrationCategory.RELATIONAL_STORE)


def test_prebuilt_base_only_platform_contract_fail_closed() -> None:
    base_only = PlatformIntegrationContract.for_provider(
        provider_id="generic_only",
        integration_kind="relational_store",
    )
    profile = IntegrationProfile(relational_store=base_only)
    with pytest.raises(TypeError, match="expected RelationalStoreIntegrationContract"):
        profile.instance_for_category(IntegrationCategory.RELATIONAL_STORE)


def test_external_work_prebuilt_canonical_accessor_requires_registry_contract() -> None:
    profile = IntegrationProfile(external_work=object())
    with pytest.raises(IntegrationContractMetadataError):
        profile.instance_for_category(IntegrationCategory.EXTERNAL_WORK)


def test_integration_profile_json_roundtrip_preserves_slug_bindings() -> None:
    register_sqlite_integration()
    profile = IntegrationProfile(relational_store=SQLITE)
    restored = IntegrationProfile.model_validate(profile.model_dump(mode="json"))
    assert restored.slug_for_category(IntegrationCategory.RELATIONAL_STORE) == SQLITE.slug
