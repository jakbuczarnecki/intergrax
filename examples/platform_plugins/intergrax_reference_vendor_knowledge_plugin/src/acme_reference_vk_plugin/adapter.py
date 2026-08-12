"""Vendor Knowledge adapter for the reference external provider."""

from __future__ import annotations

import hashlib

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
    KnowledgeVisibility,
)

from acme_reference_vk_plugin.constants import (
    ACME_COLLECTION_SCOPE_TYPE,
    ACME_DOCUMENTS_SOURCE_KIND,
    ACME_REFERENCE_PROVIDER_ID,
    ACME_STRUCTURED_RECORD_SCHEMA,
)
from acme_reference_vk_plugin.integration import AcmeReferenceWikiKnowledgeIntegration


class AcmeReferenceDocumentsKnowledgeAdapter:
    """Map bounded Acme reference documents into vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return ACME_REFERENCE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.WIKI_KNOWLEDGE

    @property
    def source_kind(self) -> str:
        return ACME_DOCUMENTS_SOURCE_KIND

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=False,
            content_fetch=True,
            binary_content=False,
            rich_text_content=False,
            structured_content=True,
            permissions=False,
            tombstones=False,
            remote_versions=True,
            reconciliation=True,
        )

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self._require_integration(integration=integration, source=source)
        collection_id = self._validate_source(source)
        return KnowledgeScopeInfo(
            source=source,
            capabilities=self.capabilities,
            safe_display_name=source.scope.safe_display_name or collection_id,
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        resolved = self._require_integration(integration=integration, source=source)
        collection_id = self._validate_source(source)
        if cursor is not None and cursor.value == "complete":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Acme reference reconciliation cursor is complete",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        documents = resolved.list_documents(collection_id=collection_id)[: max(1, limit)]
        changes = tuple(self._document_to_change(document) for document in documents)
        checkpoint = KnowledgeCursor(value="complete")
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=checkpoint,
            has_more=False,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        resolved = self._require_integration(integration=integration, source=source)
        collection_id = self._validate_source(source)
        document = resolved.get_document(
            collection_id=collection_id,
            remote_id=item.identity.remote_id,
        )
        if document is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND,
                safe_message="Acme reference document was not found",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record={
                "schema": ACME_STRUCTURED_RECORD_SCHEMA,
                "provider": self.provider_id,
                "source_kind": self.source_kind,
                "collection_id": collection_id,
                "remote_id": document.remote_id,
                "title": document.title,
                "body": document.body,
                "revision": document.revision,
            },
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        _ = integration, source, item
        return KnowledgePermissions(visibility=KnowledgeVisibility.TENANT)

    def _document_to_change(self, document) -> KnowledgeChange:
        revision = KnowledgeItemRevision(version=document.revision)
        descriptor = KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(remote_id=document.remote_id),
            revision=revision,
            title=document.title,
            item_type="article",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=document.remote_id,
            ),
            metadata={"content_hash": self._content_hash(document.body)},
        )
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            descriptor=descriptor,
            remote_id=document.remote_id,
        )

    @staticmethod
    def _content_hash(body: str) -> str:
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _validate_source(self, source: KnowledgeSourceRef) -> str:
        if source.scope.remote_scope_type != ACME_COLLECTION_SCOPE_TYPE:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Acme reference scope type is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        collection_id = source.scope.remote_scope_id.strip()
        if not collection_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Acme reference collection id is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return collection_id

    @staticmethod
    def _require_integration(
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> AcmeReferenceWikiKnowledgeIntegration:
        if not isinstance(integration, AcmeReferenceWikiKnowledgeIntegration):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Acme reference adapter requires Acme wiki integration",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return integration
