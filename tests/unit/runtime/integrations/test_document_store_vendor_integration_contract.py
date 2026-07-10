# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.document_store import (
    DocumentQueryResult,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
    PlatformIntegrationStatus,
    derive_platform_integration_id,
)
from intergrax.runtime.integrations.document_store import (
    DOCUMENT_STORE_VENDOR_INTEGRATION_CONTRACT_SCHEMA,
    DocumentStoreVendorIntegrationConfig,
    DocumentStoreVendorIntegrationContract,
    DocumentStoreVendorOperation,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_VENDOR_IMPORT_PREFIXES = (
    "pymongo",
    "cassandra",
    "boto3",
    "azure",
)


class _InMemoryDocumentStore:
    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], DocumentRecord] = {}

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._documents.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._documents[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        self._documents.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
    ) -> DocumentQueryResult:
        documents = [
            document
            for (pk, _), document in self._documents.items()
            if pk == partition_key
            and (row_key_prefix is None or document.row_key.startswith(row_key_prefix))
        ]
        documents = documents[:limit]
        return DocumentQueryResult(documents=documents, total=len(documents))

    def close(self) -> None:
        self._documents.clear()


class ExampleDocumentStoreVendorIntegration(DocumentStoreVendorIntegrationContract):
    def as_document_store(self) -> DocumentStore:
        return _InMemoryDocumentStore()


def test_document_store_vendor_contract_derives_from_platform_integration_contract() -> None:
    contract = DocumentStoreVendorIntegrationContract.for_provider(provider_id="mongodb")

    assert isinstance(contract, PlatformIntegrationContract)
    assert isinstance(contract, DocumentStoreVendorIntegrationContract)


def test_for_provider_creates_stable_category_specific_integration_identity() -> None:
    contract = DocumentStoreVendorIntegrationContract.for_provider(provider_id="mongodb")

    assert contract.schema_id == DOCUMENT_STORE_VENDOR_INTEGRATION_CONTRACT_SCHEMA
    assert contract.provider_id == "mongodb"
    assert contract.integration_kind == PlatformIntegrationKind.DOCUMENT_STORE.value
    assert contract.integration_id == derive_platform_integration_id("mongodb", "document_store")
    assert contract.integration_id == "mongodb:document_store"


def test_default_capabilities_include_read_write_health_check() -> None:
    contract = DocumentStoreVendorIntegrationContract.for_provider(provider_id="mongodb")

    assert PlatformIntegrationCapability.READ in contract.capabilities
    assert PlatformIntegrationCapability.WRITE in contract.capabilities
    assert PlatformIntegrationCapability.HEALTH_CHECK in contract.capabilities


def test_default_supported_operations_include_document_store_surface() -> None:
    contract = DocumentStoreVendorIntegrationContract.for_provider(provider_id="mongodb")

    assert contract.supported_operations == (
        DocumentStoreVendorOperation.GET,
        DocumentStoreVendorOperation.PUT,
        DocumentStoreVendorOperation.DELETE,
        DocumentStoreVendorOperation.QUERY,
        DocumentStoreVendorOperation.CLOSE,
    )


def test_public_view_includes_safe_fields_and_supported_operations() -> None:
    config = DocumentStoreVendorIntegrationConfig(
        enabled=True,
        database_name="proof_receipts",
        collection_name="receipts",
        namespace="lkw",
    )
    contract = DocumentStoreVendorIntegrationContract.for_provider(
        provider_id="mongodb",
        config=config,
    )

    public_view = contract.public_view()

    assert public_view["provider_id"] == "mongodb"
    assert public_view["integration_kind"] == PlatformIntegrationKind.DOCUMENT_STORE.value
    assert "read" in public_view["capabilities"]
    assert "write" in public_view["capabilities"]
    assert "health_check" in public_view["capabilities"]
    assert public_view["supported_operations"] == [
        "get",
        "put",
        "delete",
        "query",
        "close",
    ]
    assert public_view["config"]["database_name"] == "proof_receipts"
    assert public_view["config"]["collection_name"] == "receipts"
    assert public_view["config"]["namespace"] == "lkw"


def test_public_view_does_not_expose_secrets() -> None:
    config = DocumentStoreVendorIntegrationConfig(
        enabled=True,
        database_name="proof_receipts",
    )
    contract = DocumentStoreVendorIntegrationContract.for_provider(
        provider_id="mongodb",
        config=config,
    )

    config_view = contract.public_view()["config"]

    assert "password" not in config_view
    assert "token" not in config_view
    assert "api_key" not in config_view
    assert "uri" not in config_view
    assert "mongodb_uri" not in config_view


def test_disabled_config_health_returns_disabled() -> None:
    config = DocumentStoreVendorIntegrationConfig(enabled=False)
    contract = DocumentStoreVendorIntegrationContract.for_provider(
        provider_id="mongodb",
        config=config,
    )

    health = contract.check_health()

    assert health.status == PlatformIntegrationStatus.DISABLED
    assert health.message == "integration is disabled"


def test_as_document_store_raises_not_implemented_on_base_contract() -> None:
    contract = DocumentStoreVendorIntegrationContract.for_provider(provider_id="mongodb")

    with pytest.raises(NotImplementedError, match="as_document_store"):
        contract.as_document_store()


def test_example_vendor_subclass_returns_document_store_test_double() -> None:
    integration = ExampleDocumentStoreVendorIntegration.for_provider(provider_id="mongodb")

    store = integration.as_document_store()

    assert isinstance(store, _InMemoryDocumentStore)
    store.put(
        DocumentRecord(
            partition_key="proof_receipts/app",
            row_key="proof/kind/run-1",
            data={"proof_id": "proof-1"},
        )
    )
    document = store.get("proof_receipts/app", "proof/kind/run-1")
    assert document is not None
    assert document.data == {"proof_id": "proof-1"}


def test_no_vendor_sdk_imports_in_document_store_integration_module() -> None:
    import intergrax.runtime.integrations.document_store as document_store_module

    module_path = document_store_module.__file__
    assert module_path is not None
    source = open(module_path, encoding="utf-8").read().lower()

    for token in _FORBIDDEN_VENDOR_IMPORT_PREFIXES:
        assert f"import {token}" not in source
        assert f"from {token}" not in source
