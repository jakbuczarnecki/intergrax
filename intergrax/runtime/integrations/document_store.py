# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Document store vendor integration category contract (PROOF-RECEIPTS-1B)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Mapping

from pydantic import Field

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
    derive_platform_integration_id,
)

DOCUMENT_STORE_VENDOR_INTEGRATION_CONTRACT_SCHEMA = "document_store_vendor_integration_contract.v1"


class DocumentStoreVendorOperation(StrEnum):
    """Document store operations exposed through vendor integrations."""

    GET = "get"
    PUT = "put"
    DELETE = "delete"
    QUERY = "query"
    CLOSE = "close"


class DocumentStoreVendorKind(StrEnum):
    """Well-known document store vendor provider_id slugs — category-specific classes only."""

    MONGODB = "mongodb"
    CASSANDRA = "cassandra"
    DYNAMODB = "dynamodb"
    COSMOSDB = "cosmosdb"
    CUSTOM = "custom"


_DEFAULT_DOCUMENT_STORE_VENDOR_OPERATIONS: tuple[DocumentStoreVendorOperation, ...] = (
    DocumentStoreVendorOperation.GET,
    DocumentStoreVendorOperation.PUT,
    DocumentStoreVendorOperation.DELETE,
    DocumentStoreVendorOperation.QUERY,
    DocumentStoreVendorOperation.CLOSE,
)

_DEFAULT_DOCUMENT_STORE_VENDOR_CAPABILITIES: tuple[PlatformIntegrationCapability, ...] = (
    PlatformIntegrationCapability.READ,
    PlatformIntegrationCapability.WRITE,
    PlatformIntegrationCapability.HEALTH_CHECK,
)


class DocumentStoreVendorIntegrationConfig(PlatformIntegrationConfig):
    """Typed config for document store vendor integrations — secrets stay out of payloads."""

    database_name: str | None = None
    collection_name: str | None = None
    namespace: str | None = None


class DocumentStoreVendorIntegrationContract(PlatformIntegrationContract):
    """
    Category-specific contract for document store vendor integrations.

    Concrete vendors (MongoDB, Cassandra, DynamoDB, Cosmos DB, custom backends)
    subclass this type — one integration class per category. The same provider_id
    may appear in other categories through separate integration classes.

    Implements the provider-neutral DocumentStore surface via as_document_store().
    """

    schema_id: Literal["document_store_vendor_integration_contract.v1"] = (
        DOCUMENT_STORE_VENDOR_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.DOCUMENT_STORE.value
    supported_operations: tuple[DocumentStoreVendorOperation, ...] = Field(
        default_factory=lambda: _DEFAULT_DOCUMENT_STORE_VENDOR_OPERATIONS
    )
    config: DocumentStoreVendorIntegrationConfig = Field(
        default_factory=DocumentStoreVendorIntegrationConfig
    )

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        supported_operations: tuple[DocumentStoreVendorOperation, ...] = _DEFAULT_DOCUMENT_STORE_VENDOR_OPERATIONS,
        capabilities: tuple[PlatformIntegrationCapability, ...] = _DEFAULT_DOCUMENT_STORE_VENDOR_CAPABILITIES,
        display_name: str | None = None,
        version: str | None = None,
        config: DocumentStoreVendorIntegrationConfig | None = None,
    ) -> DocumentStoreVendorIntegrationContract:
        if config is None:
            config_field = cls.model_fields["config"]
            if config_field.default_factory is not None:
                config = config_field.default_factory()
            elif isinstance(config_field.default, DocumentStoreVendorIntegrationConfig):
                config = config_field.default
            else:
                config = DocumentStoreVendorIntegrationConfig()
        return cls(
            integration_id=derive_platform_integration_id(
                provider_id,
                PlatformIntegrationKind.DOCUMENT_STORE.value,
            ),
            provider_id=provider_id,
            display_name=display_name,
            version=version,
            capabilities=capabilities,
            supported_operations=supported_operations,
            config=config,
        )

    def as_document_store(self) -> DocumentStore:
        """Return the provider-neutral DocumentStore surface — override in concrete integrations."""
        raise NotImplementedError(
            f"{type(self).__name__} must override as_document_store() for vendor I/O"
        )

    def public_view(self) -> Mapping[str, Any]:
        view = dict(super().public_view())
        view["supported_operations"] = [operation.value for operation in self.supported_operations]
        return view
