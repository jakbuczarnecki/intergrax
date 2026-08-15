"""Indexed materializer for the reference external provider."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    MaterializedConnectedSourceDocument,
    build_materialized_connected_source_document,
    validate_materializer_source,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity

from acme_reference_vk_plugin.constants import (
    ACME_DOCUMENTS_SOURCE_KIND,
    ACME_INDEXED_RUNTIME_REF,
    ACME_REFERENCE_PROVIDER_ID,
    ACME_STRUCTURED_RECORD_SCHEMA,
)

_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=ACME_REFERENCE_PROVIDER_ID,
    integration_category=IntegrationCategory.WIKI_KNOWLEDGE,
    source_kind=ACME_DOCUMENTS_SOURCE_KIND,
)


class AcmeReferenceDocumentMaterializer:
    identity = _IDENTITY
    runtime_ref = ACME_INDEXED_RUNTIME_REF
    schema_name = ACME_STRUCTURED_RECORD_SCHEMA

    def materialize(
        self,
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
        revision: KnowledgeItemRevision | None,
        permissions: KnowledgePermissions | None,
    ) -> MaterializedConnectedSourceDocument:
        _ = permissions
        validate_materializer_source(self.identity, source)
        record = content.structured_record or {}
        title = str(record.get("title") or "Acme reference document")
        body = str(record.get("body") or "")
        markdown = "\n".join(
            [
                f"# {title}",
                "",
                body,
                "",
                f"Provider: {self.identity.provider_id}",
                f"Source kind: {self.identity.source_kind}",
                f"Remote id: {remote_id}",
            ]
        )
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=f"acme-reference-{remote_id}.md",
            revision=revision,
            permissions=permissions,
        )
