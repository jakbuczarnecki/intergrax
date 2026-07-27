# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DocumentStore-backed repository for knowledge source bindings."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingAlreadyExists,
    KnowledgeSourceBindingCorruptRecord,
    KnowledgeSourceBindingNotFound,
    KnowledgeSourceBindingStatus,
    KnowledgeSourceBindingVersionConflict,
)

_PARTITION_PREFIX = "vendor_knowledge_bindings"
_ROW_PREFIX = "binding"
_MAX_BINDING_LIST_SCAN = 10_000
_FORBIDDEN_SECRET_FIELDS: frozenset[str] = frozenset(
    {
        "access_token",
        "refresh_token",
        "api_key",
        "password",
        "client_secret",
        "authorization_header",
        "signed_download_url",
    }
)


def binding_partition_key(tenant_id: str) -> str:
    cleaned = tenant_id.strip()
    if not cleaned:
        raise ValueError("tenant_id must be a non-empty string")
    return f"{_PARTITION_PREFIX}:{cleaned}"


def binding_row_key(binding_id: str) -> str:
    cleaned = binding_id.strip()
    if not cleaned:
        raise ValueError("binding_id must be a non-empty string")
    return f"{_ROW_PREFIX}:{cleaned}"


def binding_to_document(binding: KnowledgeSourceBinding) -> DocumentRecord:
    data = binding.model_dump(mode="json")
    for key in _FORBIDDEN_SECRET_FIELDS:
        if key in data:
            raise KnowledgeSourceBindingCorruptRecord(
                "binding document must not contain secret-bearing fields"
            )
    return DocumentRecord(
        partition_key=binding_partition_key(binding.tenant_id),
        row_key=binding_row_key(binding.binding_id),
        data=data,
        ttl_seconds=None,
    )


def binding_from_document(document: DocumentRecord) -> KnowledgeSourceBinding:
    data: dict[str, Any] = dict(document.data)
    for key in _FORBIDDEN_SECRET_FIELDS:
        if key in data:
            raise KnowledgeSourceBindingCorruptRecord(
                "binding document must not contain secret-bearing fields"
            )

    tenant_id = data.get("tenant_id")
    binding_id = data.get("binding_id")
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise KnowledgeSourceBindingCorruptRecord("binding document tenant_id is invalid")
    if not isinstance(binding_id, str) or not binding_id.strip():
        raise KnowledgeSourceBindingCorruptRecord("binding document binding_id is invalid")

    expected_partition = binding_partition_key(tenant_id)
    if document.partition_key != expected_partition:
        raise KnowledgeSourceBindingCorruptRecord(
            "binding document partition does not match tenant"
        )

    expected_row = binding_row_key(binding_id)
    if document.row_key != expected_row:
        raise KnowledgeSourceBindingCorruptRecord(
            "binding document row key does not match binding_id"
        )

    try:
        return KnowledgeSourceBinding.model_validate(data)
    except ValidationError as exc:
        raise KnowledgeSourceBindingCorruptRecord(
            "binding document payload is invalid"
        ) from exc


class DocumentStoreKnowledgeSourceBindingRepository:
    """Production repository mapping bindings onto a provider-neutral DocumentStore."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._document_store = document_store

    def create(self, binding: KnowledgeSourceBinding) -> None:
        partition_key = binding_partition_key(binding.tenant_id)
        row_key = binding_row_key(binding.binding_id)
        existing = self._document_store.get(partition_key, row_key)
        if existing is not None:
            raise KnowledgeSourceBindingAlreadyExists(
                "knowledge source binding already exists"
            )
        self._document_store.put(binding_to_document(binding))

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSourceBinding | None:
        cleaned_tenant = tenant_id.strip()
        cleaned_binding = binding_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if not cleaned_binding:
            raise ValueError("binding_id must be a non-empty string")

        document = self._document_store.get(
            binding_partition_key(cleaned_tenant),
            binding_row_key(cleaned_binding),
        )
        if document is None:
            return None
        binding = binding_from_document(document)
        if binding.tenant_id != cleaned_tenant:
            raise KnowledgeSourceBindingCorruptRecord(
                "binding document tenant does not match lookup tenant"
            )
        return binding

    def update(
        self,
        binding: KnowledgeSourceBinding,
        *,
        expected_configuration_version: int,
    ) -> None:
        existing = self.get(tenant_id=binding.tenant_id, binding_id=binding.binding_id)
        if existing is None:
            raise KnowledgeSourceBindingNotFound("knowledge source binding was not found")

        if existing.configuration_version != expected_configuration_version:
            raise KnowledgeSourceBindingVersionConflict(
                "knowledge source binding configuration version conflict"
            )
        if binding.configuration_version != expected_configuration_version + 1:
            raise KnowledgeSourceBindingVersionConflict(
                "knowledge source binding configuration version conflict"
            )

        if (
            existing.binding_id != binding.binding_id
            or existing.tenant_id != binding.tenant_id
            or existing.provider_id != binding.provider_id
            or existing.integration_kind != binding.integration_kind
            or existing.source_kind != binding.source_kind
        ):
            raise KnowledgeSourceBindingVersionConflict(
                "knowledge source binding identity fields are immutable"
            )

        self._document_store.put(binding_to_document(binding))

    def list(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        status: KnowledgeSourceBindingStatus | None = None,
    ) -> tuple[KnowledgeSourceBinding, ...]:
        cleaned_tenant = tenant_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        result = self._document_store.query(
            binding_partition_key(cleaned_tenant),
            limit=_MAX_BINDING_LIST_SCAN + 1,
            row_key_prefix=f"{_ROW_PREFIX}:",
        )
        if len(result.documents) > _MAX_BINDING_LIST_SCAN:
            raise KnowledgeSourceBindingCorruptRecord(
                "binding list exceeds scan limit"
            )
        if result.total > _MAX_BINDING_LIST_SCAN:
            raise KnowledgeSourceBindingCorruptRecord(
                "binding list exceeds scan limit"
            )

        bindings: list[KnowledgeSourceBinding] = []
        for document in result.documents:
            binding = binding_from_document(document)
            if binding.tenant_id != cleaned_tenant:
                raise KnowledgeSourceBindingCorruptRecord(
                    "binding document tenant does not match lookup tenant"
                )
            if status is not None and binding.status is not status:
                continue
            bindings.append(binding)
        bindings.sort(key=lambda item: item.binding_id)
        return tuple(bindings[:limit])
