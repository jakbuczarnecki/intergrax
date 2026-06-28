# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from intergrax.runtime.integrations.contracts import (
    PLATFORM_INTEGRATION_CONTRACT_SCHEMA,
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
    PlatformIntegrationSecurityPosture,
    PlatformIntegrationStatus,
    derive_platform_integration_id,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_VENDOR_IMPORT_PREFIXES = (
    "langfuse",
    "arize",
    "phoenix",
    "opentelemetry",
    "elasticsearch",
)


class ExampleSearchIntegration(PlatformIntegrationContract):
    integration_kind: str = PlatformIntegrationKind.SEARCH.value


class ExampleVectorStoreIntegration(PlatformIntegrationContract):
    integration_kind: str = PlatformIntegrationKind.VECTOR_STORE.value


class SensitiveIntegrationConfig(PlatformIntegrationConfig):
    api_key: str | None = None
    token: str | None = None


def test_platform_integration_contract_exposes_core_fields() -> None:
    contract = PlatformIntegrationContract.for_provider(
        provider_id="elasticsearch",
        integration_kind=PlatformIntegrationKind.OBSERVABILITY_VENDOR,
        capabilities=(PlatformIntegrationCapability.EXPORT, PlatformIntegrationCapability.HEALTH_CHECK),
        display_name="Elasticsearch Observability",
        version="1.0.0",
    )

    assert contract.schema_id == PLATFORM_INTEGRATION_CONTRACT_SCHEMA
    assert contract.integration_id == "elasticsearch:observability_vendor"
    assert contract.provider_id == "elasticsearch"
    assert contract.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    assert PlatformIntegrationCapability.EXPORT in contract.capabilities
    assert PlatformIntegrationCapability.HEALTH_CHECK in contract.capabilities


def test_platform_integration_config_disabled_by_default() -> None:
    config = PlatformIntegrationConfig()

    assert config.enabled is False


def test_same_provider_id_with_two_integration_kinds() -> None:
    observability = PlatformIntegrationContract.for_provider(
        provider_id="elasticsearch",
        integration_kind=PlatformIntegrationKind.OBSERVABILITY_VENDOR,
        capabilities=(PlatformIntegrationCapability.EXPORT,),
    )
    vector_store = PlatformIntegrationContract.for_provider(
        provider_id="elasticsearch",
        integration_kind=PlatformIntegrationKind.VECTOR_STORE,
        capabilities=(PlatformIntegrationCapability.READ, PlatformIntegrationCapability.WRITE),
    )

    assert observability.provider_id == vector_store.provider_id == "elasticsearch"
    assert observability.integration_kind != vector_store.integration_kind


def test_distinct_integration_identities_for_same_provider_different_kind() -> None:
    search = PlatformIntegrationContract.for_provider(
        provider_id="elasticsearch",
        integration_kind=PlatformIntegrationKind.SEARCH,
    )
    vector_store = PlatformIntegrationContract.for_provider(
        provider_id="elasticsearch",
        integration_kind=PlatformIntegrationKind.VECTOR_STORE,
    )

    assert search.integration_id == derive_platform_integration_id("elasticsearch", "search")
    assert vector_store.integration_id == derive_platform_integration_id("elasticsearch", "vector_store")
    assert search.integration_id != vector_store.integration_id


def test_security_posture_does_not_expose_secrets_by_default() -> None:
    config = SensitiveIntegrationConfig(
        enabled=True,
        timeout_seconds=5.0,
        api_key="super-secret-key",
        token="raw-token",
    )
    contract = PlatformIntegrationContract.for_provider(
        provider_id="example",
        integration_kind=PlatformIntegrationKind.NOTIFICATION,
        config=config,
    )

    public_view = contract.public_view()
    serialized = json.dumps(public_view)

    assert contract.security_posture.expose_secrets is False
    assert contract.security_posture.expose_raw_payloads is False
    assert "api_key" not in public_view["config"]
    assert "token" not in public_view["config"]
    assert "super-secret-key" not in serialized
    assert "raw-token" not in serialized


def test_example_subclass_derives_from_platform_integration_contract() -> None:
    integration = ExampleSearchIntegration(
        integration_id=derive_platform_integration_id("elasticsearch", PlatformIntegrationKind.SEARCH.value),
        provider_id="elasticsearch",
        capabilities=(PlatformIntegrationCapability.READ,),
    )

    assert isinstance(integration, PlatformIntegrationContract)
    assert integration.integration_kind == PlatformIntegrationKind.SEARCH.value


def test_generic_contract_has_no_observability_vendor_behavior() -> None:
    contract = PlatformIntegrationContract.for_provider(
        provider_id="generic",
        integration_kind=PlatformIntegrationKind.STORAGE,
    )

    assert not hasattr(contract, "export")
    assert not hasattr(contract, "export_envelope")
    assert contract.check_health().status is PlatformIntegrationStatus.DISABLED


def test_no_vendor_sdk_imports_in_contract_module() -> None:
    import intergrax.runtime.integrations.contracts as contracts_module

    module_path = contracts_module.__file__
    assert module_path is not None
    source = open(module_path, encoding="utf-8").read().lower()

    for token in _FORBIDDEN_VENDOR_IMPORT_PREFIXES:
        assert token not in source
