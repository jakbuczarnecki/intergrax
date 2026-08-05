# © Artur Czarnecki. All rights reserved.

from intergrax.knowledge.contracts import KnowledgeDocument


def knowledge_document(
    content: str,
    *,
    tenant_id: str = "tenant-a",
    namespace: str | None = None,
    workspace_id: str | None = None,
    document_id: str = "doc-1",
    metadata: dict[str, object] | None = None,
) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "workspace_id": workspace_id,
            },
            "content": content,
            "metadata": metadata or {},
            "provenance": {
                "source_kind": "graph-test",
                "source_id": f"source:{document_id}",
            },
        }
    )
