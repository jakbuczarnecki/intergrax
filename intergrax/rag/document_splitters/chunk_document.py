# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from intergrax.knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeDocumentIdentity,
    KnowledgeDocumentProvenance,
)
from intergrax.knowledge.contracts.document import knowledge_metadata_to_plain
from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _derive_chunk_document_id(
    *,
    source_document_id: str,
    strategy_id: str,
    chunk_index: int,
    content: str,
) -> str:
    content_hash = _sha256_hex(content)
    id_material = f"{source_document_id}|{strategy_id}|{chunk_index}|{content_hash}"
    return _sha256_hex(id_material)[:32]


def build_derived_chunk(
    source_document: KnowledgeDocument,
    *,
    content: str,
    strategy_id: str,
    chunk_index: int,
    metadata_updates: Mapping[str, JsonValue] | None = None,
) -> KnowledgeDocument:
    if not content.strip():
        raise ValueError("chunk content must be a non-empty string")

    derived_id = _derive_chunk_document_id(
        source_document_id=source_document.identity.document_id,
        strategy_id=strategy_id,
        chunk_index=chunk_index,
        content=content,
    )
    content_hash = _sha256_hex(content)

    metadata = dict(knowledge_metadata_to_plain(source_document.metadata))
    metadata[ChunkMetadataKey.CHUNK_ID.value] = derived_id
    metadata[ChunkMetadataKey.CHUNK_INDEX.value] = chunk_index
    metadata[ChunkMetadataKey.CHUNK_STRATEGY.value] = strategy_id
    metadata[ChunkMetadataKey.CHUNK_SIZE.value] = len(content)

    if metadata_updates:
        metadata.update(metadata_updates)

    source_provenance = source_document.provenance
    provenance = KnowledgeDocumentProvenance(
        source_kind=source_provenance.source_kind,
        source_id=source_provenance.source_id,
        source_parent_id=source_provenance.source_parent_id,
        provider_id=source_provenance.provider_id,
        source_revision=source_provenance.source_revision,
        source_uri=source_provenance.source_uri,
        content_hash=content_hash,
    )

    return KnowledgeDocument(
        schema_version=source_document.schema_version,
        identity=KnowledgeDocumentIdentity(
            document_id=derived_id,
            root_document_id=source_document.identity.root_document_id,
            parent_document_id=source_document.identity.document_id,
        ),
        scope=source_document.scope,
        content=content,
        metadata=metadata,
        provenance=provenance,
    )
