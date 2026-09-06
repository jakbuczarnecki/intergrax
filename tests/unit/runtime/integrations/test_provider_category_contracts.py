# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.runtime.integrations.categories import (
    OBSERVABILITY_BACKEND_CATEGORY,
    OBSERVABILITY_VENDOR_INTEGRATION_KIND,
    PROVIDER_CATEGORY_CONTRACT_REGISTRY,
    VectorStoreIntegrationContract,
)
from intergrax.runtime.integrations.document_store import DocumentStoreVendorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
    derive_platform_integration_id,
)
from intergrax.runtime.integrations.observability import ObservabilityVendorIntegrationContract

pytestmark = pytest.mark.unit

_FORBIDDEN_VENDOR_IMPORT_PREFIXES = (
    "langfuse",
    "arize",
    "phoenix",
    "opentelemetry",
    "elasticsearch",
)


class SensitiveCategoryConfig(PlatformIntegrationConfig):
    api_key: str | None = None
    token: str | None = None


def _unique_layout_categories() -> frozenset[str]:
    return frozenset(SLUG_CATEGORY.values())


# P2-002-B1: typed contract registered before first-party provider layout folders (B2).
_REGISTRY_CATEGORIES_PENDING_LAYOUT: frozenset[str] = frozenset()


def test_every_layout_category_has_contract_or_alias() -> None:
    layout_categories = _unique_layout_categories()
    registry_categories = frozenset(PROVIDER_CATEGORY_CONTRACT_REGISTRY.keys())
    assert layout_categories <= registry_categories
    pending = registry_categories - layout_categories
    assert pending <= _REGISTRY_CATEGORIES_PENDING_LAYOUT


def test_every_category_contract_derives_from_platform_integration_contract() -> None:
    for contract_cls in PROVIDER_CATEGORY_CONTRACT_REGISTRY.values():
        assert issubclass(contract_cls, PlatformIntegrationContract)


def test_observability_backend_aligns_with_observability_vendor_contract() -> None:
    contract_cls = PROVIDER_CATEGORY_CONTRACT_REGISTRY[OBSERVABILITY_BACKEND_CATEGORY]

    assert contract_cls is ObservabilityVendorIntegrationContract
    assert OBSERVABILITY_BACKEND_CATEGORY == PlatformIntegrationKind.OBSERVABILITY_BACKEND.value
    assert OBSERVABILITY_VENDOR_INTEGRATION_KIND == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value

    contract = ObservabilityVendorIntegrationContract.for_provider(provider_id="langfuse")
    assert contract.integration_kind == OBSERVABILITY_VENDOR_INTEGRATION_KIND
    assert contract.integration_kind != OBSERVABILITY_BACKEND_CATEGORY


def test_category_contracts_have_stable_schema_id() -> None:
    for category, contract_cls in PROVIDER_CATEGORY_CONTRACT_REGISTRY.items():
        if contract_cls in (
            ObservabilityVendorIntegrationContract,
            DocumentStoreVendorIntegrationContract,
        ):
            continue
        instance = contract_cls.for_provider(provider_id=f"example_{category}")
        assert instance.schema_id.endswith("_integration_contract.v1")
        assert instance.schema_id == f"{category}_integration_contract.v1"


def test_category_contracts_use_matching_integration_kind() -> None:
    for category, contract_cls in PROVIDER_CATEGORY_CONTRACT_REGISTRY.items():
        if contract_cls is ObservabilityVendorIntegrationContract:
            contract = contract_cls.for_provider(provider_id="otel")
            assert contract.integration_kind == OBSERVABILITY_VENDOR_INTEGRATION_KIND
            continue
        if contract_cls is DocumentStoreVendorIntegrationContract:
            contract = contract_cls.for_provider(provider_id="mongodb")
            assert contract.integration_kind == PlatformIntegrationKind.DOCUMENT_STORE.value
            continue
        contract = contract_cls.for_provider(provider_id=f"example_{category}")
        assert contract.integration_kind == category


def test_category_contract_config_disabled_by_default() -> None:
    for contract_cls in PROVIDER_CATEGORY_CONTRACT_REGISTRY.values():
        contract = contract_cls.for_provider(provider_id="example")
        assert contract.config.enabled is False


def test_category_contract_public_view_does_not_expose_secrets() -> None:
    config = SensitiveCategoryConfig(
        enabled=True,
        api_key="super-secret-key",
        token="raw-token",
    )
    contract = VectorStoreIntegrationContract.for_provider(provider_id="pinecone", config=config)
    public_view = contract.public_view()
    serialized = json.dumps(public_view)

    assert "api_key" not in public_view["config"]
    assert "token" not in public_view["config"]
    assert "super-secret-key" not in serialized
    assert "raw-token" not in serialized


def test_same_provider_id_distinct_integration_ids_across_categories() -> None:
    observability = ObservabilityVendorIntegrationContract.for_provider(provider_id="elasticsearch")
    vector_store = VectorStoreIntegrationContract.for_provider(provider_id="elasticsearch")

    assert observability.provider_id == vector_store.provider_id == "elasticsearch"
    assert observability.integration_id == derive_platform_integration_id(
        "elasticsearch",
        OBSERVABILITY_VENDOR_INTEGRATION_KIND,
    )
    assert vector_store.integration_id == derive_platform_integration_id(
        "elasticsearch",
        PlatformIntegrationKind.VECTOR_STORE.value,
    )
    assert observability.integration_id != vector_store.integration_id


def test_no_concrete_provider_classes_in_category_modules() -> None:
    import intergrax.runtime.integrations.categories as categories_pkg

    for module_name in categories_pkg.__all__:
        if not module_name.endswith("IntegrationContract"):
            continue
        exported = getattr(categories_pkg, module_name)
        assert exported.__name__.endswith("IntegrationContract")


def test_no_vendor_sdk_imports_in_category_modules() -> None:
    import importlib
    import pkgutil

    import intergrax.runtime.integrations.categories as categories_pkg

    for module_info in pkgutil.iter_modules(categories_pkg.__path__, categories_pkg.__name__ + "."):
        module = importlib.import_module(module_info.name)
        module_path = module.__file__
        if module_path is None or not module_path.endswith(".py"):
            continue
        source = open(module_path, encoding="utf-8").read().lower()
        for token in _FORBIDDEN_VENDOR_IMPORT_PREFIXES:
            assert token not in source
