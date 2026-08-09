# © Artur Czarnecki. All rights reserved.

"""Provider-neutral boundary for indexed connected-source materialization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from intergrax.knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeDocumentIdentity,
    KnowledgeDocumentProvenance,
    KnowledgeDocumentScope,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity


class VendorKnowledgeMaterializationError(ValueError):
    """Safe provider-neutral materialization failure."""


@dataclass(frozen=True, slots=True)
class MaterializedConnectedSourceDocument:
    knowledge_document: KnowledgeDocument
    logical_source_path: str
    safe_file_name: str
    markdown: str
    content_hash: str
    document_id: str
    source_revision: KnowledgeItemRevision | None


def validate_materializer_source(
    identity: VendorKnowledgeSourceIdentity,
    source: KnowledgeSourceRef,
) -> None:
    actual = VendorKnowledgeSourceIdentity(
        provider_id=source.provider_id,
        integration_category=source.integration_kind,
        source_kind=source.source_kind,
    )
    if actual != identity:
        raise VendorKnowledgeMaterializationError(
            "connected_source_materializer_identity_mismatch"
        )


def build_materialized_connected_source_document(
    *,
    identity: VendorKnowledgeSourceIdentity,
    source: KnowledgeSourceRef,
    tenant_id: str,
    workspace_id: str,
    binding_id: str,
    source_id: str,
    remote_id: str,
    markdown: str,
    safe_file_name: str,
    revision: KnowledgeItemRevision | None,
    permissions: KnowledgePermissions | None,
) -> MaterializedConnectedSourceDocument:
    markdown_bytes = markdown.encode("utf-8")
    content_hash = hashlib.sha256(markdown_bytes).hexdigest()
    document_id = _connected_document_id(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        provider_id=identity.provider_id,
        integration_kind=identity.integration_category.value,
        source_kind=identity.source_kind,
        binding_id=binding_id,
        remote_id=remote_id,
    )
    revision_token = None
    if revision is not None:
        revision_token = hashlib.sha256(
            json.dumps(
                revision.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    metadata: dict[str, object] = {
        "vendor_knowledge_provider_id": identity.provider_id,
        "vendor_knowledge_integration_kind": identity.integration_category.value,
        "vendor_knowledge_source_kind": identity.source_kind,
        "vendor_knowledge_connection_ref": source.connection_ref,
        "vendor_knowledge_binding_id": binding_id,
        "vendor_knowledge_remote_id": remote_id,
    }
    if permissions is not None:
        metadata["permissions"] = permissions.model_dump(mode="json")
    document = KnowledgeDocument(
        schema_version=1,
        identity=KnowledgeDocumentIdentity(
            document_id=document_id,
            root_document_id=document_id,
        ),
        scope=KnowledgeDocumentScope(
            tenant_id=tenant_id,
            namespace=binding_id,
            workspace_id=workspace_id,
        ),
        content=markdown,
        metadata=metadata,
        provenance=KnowledgeDocumentProvenance(
            source_kind=identity.source_kind,
            source_id=remote_id,
            source_parent_id=source_id,
            provider_id=identity.provider_id,
            source_revision=revision_token,
            content_hash=content_hash,
        ),
    )
    return MaterializedConnectedSourceDocument(
        knowledge_document=document,
        logical_source_path=_connected_logical_path(
            source_id=source_id,
            remote_id=remote_id,
            source_kind=identity.source_kind,
        ),
        safe_file_name=safe_file_name,
        markdown=markdown,
        content_hash=content_hash,
        document_id=document_id,
        source_revision=revision,
    )


def _connected_logical_path(*, source_id: str, remote_id: str, source_kind: str) -> str:
    payload = json.dumps(
        {"source_id": source_id.strip(), "remote_id": remote_id.strip()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    safe_kind = re.sub(r"[^a-z0-9_-]+", "-", source_kind.strip().lower()).strip("-")
    if not safe_kind:
        raise VendorKnowledgeMaterializationError("source_kind_required")
    digest = hashlib.sha256(payload).hexdigest()
    return f"connected/{safe_kind}-message/{digest}.md"


def _connected_document_id(
    *,
    tenant_id: str,
    workspace_id: str,
    provider_id: str,
    integration_kind: str,
    source_kind: str,
    binding_id: str,
    remote_id: str,
) -> str:
    payload = json.dumps(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "provider_id": provider_id.strip(),
            "integration_kind": integration_kind.strip(),
            "source_kind": source_kind.strip(),
            "binding_id": binding_id.strip(),
            "remote_id": remote_id.strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"lkwdoc:{hashlib.sha256(payload).hexdigest()[:32]}"
