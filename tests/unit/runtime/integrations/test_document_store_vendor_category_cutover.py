# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
import pkgutil
import re

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.cassandra.integration import (
    CASSANDRA_DOCUMENT_STORE_PROVIDER_ID,
    CassandraDocumentStoreIntegration,
)
from intergrax.integrations.providers.document_store.dynamodb.integration import (
    DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
    DynamoDBDocumentStoreIntegration,
    DynamodbDocumentStoreIntegration,
)
from intergrax.integrations.providers.document_store.mongodb import MongoDBDocumentStore
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    MongoDBDocumentStoreIntegration,
    MongodbDocumentStoreIntegration,
)
from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationKind,
    derive_platform_integration_id,
)
from intergrax.runtime.integrations.document_store import (
    DOCUMENT_STORE_VENDOR_INTEGRATION_CONTRACT_SCHEMA,
    DocumentStoreVendorIntegrationConfig,
    DocumentStoreVendorIntegrationContract,
    DocumentStoreVendorOperation,
)

pytestmark = pytest.mark.unit

_ACTIVE_VENDOR_INTEGRATIONS = (
    MongoDBDocumentStoreIntegration,
    CassandraDocumentStoreIntegration,
    DynamoDBDocumentStoreIntegration,
)

_PROVIDER_MODULES = (
    "intergrax.integrations.providers.document_store.mongodb.integration",
    "intergrax.integrations.providers.document_store.cassandra.integration",
    "intergrax.integrations.providers.document_store.dynamodb.integration",
)


def test_document_store_registry_uses_vendor_contract() -> None:
    contract_cls = PROVIDER_CATEGORY_CONTRACT_REGISTRY["document_store"]
    assert contract_cls is DocumentStoreVendorIntegrationContract


def test_old_document_store_contract_not_importable_from_categories() -> None:
    import intergrax.runtime.integrations.categories as categories_pkg

    assert not hasattr(categories_pkg, "DocumentStoreIntegrationContract")


def test_active_vendor_integrations_inherit_vendor_contract() -> None:
    for integration_cls in _ACTIVE_VENDOR_INTEGRATIONS:
        assert issubclass(integration_cls, DocumentStoreVendorIntegrationContract)


def test_active_vendor_integrations_do_not_use_removed_category_contract() -> None:
    for module_name in _PROVIDER_MODULES:
        module = importlib.import_module(module_name)
        source = open(module.__file__, encoding="utf-8").read()
        assert "DocumentStoreIntegrationContract" not in source


def test_mongodb_integration_identity_and_operations() -> None:
    integration = MongoDBDocumentStoreIntegration.for_provider(provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID)

    assert integration.provider_id == "mongodb"
    assert integration.integration_id == "mongodb:document_store"
    assert integration.integration_kind == PlatformIntegrationKind.DOCUMENT_STORE.value
    assert integration.schema_id == DOCUMENT_STORE_VENDOR_INTEGRATION_CONTRACT_SCHEMA
    assert integration.supported_operations == (
        DocumentStoreVendorOperation.GET,
        DocumentStoreVendorOperation.PUT,
        DocumentStoreVendorOperation.DELETE,
        DocumentStoreVendorOperation.QUERY,
        DocumentStoreVendorOperation.CLOSE,
    )
    assert PlatformIntegrationCapability.READ in integration.capabilities
    assert PlatformIntegrationCapability.WRITE in integration.capabilities
    assert PlatformIntegrationCapability.HEALTH_CHECK in integration.capabilities


def test_mongodb_as_document_store_requires_client() -> None:
    integration = MongoDBDocumentStoreIntegration.for_provider(provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID)

    with pytest.raises(IntegrationConfigurationError, match="requires a catalog client"):
        integration.as_document_store()


def test_mongodb_public_view_excludes_secrets() -> None:
    from intergrax.integrations.providers.document_store.mongodb.integration import (
        MongoDBDocumentStoreIntegrationConfig,
    )

    integration = MongoDBDocumentStoreIntegration.for_provider(
        provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
        config=MongoDBDocumentStoreIntegrationConfig(
            enabled=True,
            database_name="proof_receipts",
            collection_name="receipts",
        ),
    )
    config_view = integration.public_view()["config"]

    for forbidden in ("uri", "mongodb_uri", "password", "credentials", "api_key", "token"):
        assert forbidden not in config_view


def test_mongodb_lazy_export_resolves_document_store() -> None:
    from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore

    assert MongoDBDocumentStore is _MongoDBDocumentStore


def test_mongodb_compatibility_alias_points_to_canonical_class() -> None:
    assert MongodbDocumentStoreIntegration is MongoDBDocumentStoreIntegration


def test_cassandra_integration_as_document_store_and_identity() -> None:
    integration = CassandraDocumentStoreIntegration.for_provider(
        provider_id=CASSANDRA_DOCUMENT_STORE_PROVIDER_ID,
    )

    assert integration.integration_id == derive_platform_integration_id("cassandra", "document_store")
    assert issubclass(integration.__class__, DocumentStoreVendorIntegrationContract)

    with pytest.raises(IntegrationConfigurationError, match="requires a catalog client"):
        integration.as_document_store()


def test_dynamodb_integration_as_document_store_and_identity() -> None:
    integration = DynamoDBDocumentStoreIntegration.for_provider(
        provider_id=DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
    )

    assert integration.integration_id == derive_platform_integration_id("dynamodb", "document_store")
    assert issubclass(integration.__class__, DocumentStoreVendorIntegrationContract)
    assert DynamodbDocumentStoreIntegration is DynamoDBDocumentStoreIntegration

    with pytest.raises(IntegrationConfigurationError, match="requires a catalog client"):
        integration.as_document_store()


def test_vendor_configs_derive_from_document_store_vendor_config() -> None:
    from intergrax.integrations.providers.document_store.cassandra.integration import (
        CassandraDocumentStoreIntegrationConfig,
    )
    from intergrax.integrations.providers.document_store.dynamodb.integration import (
        DynamoDBDocumentStoreIntegrationConfig,
    )
    from intergrax.integrations.providers.document_store.mongodb.integration import (
        MongoDBDocumentStoreIntegrationConfig,
    )

    for config_cls in (
        MongoDBDocumentStoreIntegrationConfig,
        CassandraDocumentStoreIntegrationConfig,
        DynamoDBDocumentStoreIntegrationConfig,
    ):
        assert issubclass(config_cls, DocumentStoreVendorIntegrationConfig)


def _module_imports_vendor_sdk(source: str, token: str) -> bool:
    pattern = re.compile(
        rf"(^|\n)\s*(import {re.escape(token)}\b|from {re.escape(token)}(\.| import))"
    )
    return pattern.search(source) is not None


def test_vendor_sdk_imports_stay_inside_provider_packages() -> None:
    opens_expectations = {
        "intergrax.integrations.providers.document_store.mongodb": "pymongo",
        "intergrax.integrations.providers.document_store.cassandra": "cassandra",
    }
    for package_name, token in opens_expectations.items():
        package = importlib.import_module(package_name)
        found_in_opens = False
        for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            module = importlib.import_module(module_info.name)
            if module.__file__ is None:
                continue
            source = open(module.__file__, encoding="utf-8").read()
            if _module_imports_vendor_sdk(source, token):
                if module_info.name.endswith(".opens"):
                    found_in_opens = True
                else:
                    pytest.fail(f"{token} import leaked outside opens in {module_info.name}")
        assert found_in_opens, f"expected {token} usage in {package_name} opens module"

    dynamodb_package = importlib.import_module(
        "intergrax.integrations.providers.document_store.dynamodb",
    )
    for module_info in pkgutil.walk_packages(
        dynamodb_package.__path__,
        dynamodb_package.__name__ + ".",
    ):
        module = importlib.import_module(module_info.name)
        if module.__file__ is None:
            continue
        source = open(module.__file__, encoding="utf-8").read()
        assert not _module_imports_vendor_sdk(source, "boto3")


def test_as_document_store_returns_document_store_protocol() -> None:
    class _StubStore:
        def get(self, partition_key: str, row_key: str):
            return None

        def put(self, document: object) -> None:
            return None

        def delete(self, partition_key: str, row_key: str) -> None:
            return None

        def query(self, partition_key: str, *, limit: int = 100, row_key_prefix: str | None = None):
            return None

        def close(self) -> None:
            return None

    stub: DocumentStore = _StubStore()
    integration = MongoDBDocumentStoreIntegration.from_client(stub)
    assert integration.as_document_store() is stub
