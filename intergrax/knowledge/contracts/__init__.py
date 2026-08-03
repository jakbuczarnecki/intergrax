# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public knowledge document contract exports."""

from intergrax.knowledge.contracts.document import (
    KnowledgeDocument,
    KnowledgeDocumentIdentity,
    KnowledgeDocumentProvenance,
    KnowledgeDocumentScope,
    dump_knowledge_document,
    load_knowledge_document,
)

__all__ = [
    "KnowledgeDocument",
    "KnowledgeDocumentIdentity",
    "KnowledgeDocumentProvenance",
    "KnowledgeDocumentScope",
    "dump_knowledge_document",
    "load_knowledge_document",
]
