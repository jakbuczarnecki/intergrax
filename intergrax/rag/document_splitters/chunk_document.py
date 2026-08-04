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

_CANONICAL_CHUNK_METADATA_KEYS = frozenset(
    {
        ChunkMetadataKey.CHUNK_ID.value,
        ChunkMetadataKey.CHUNK_INDEX.value,
        ChunkMetadataKey.CHUNK_STRATEGY.value,
        ChunkMetadataKey.CHUNK_SIZE.value,
    }
)


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


def validate_derived_chunk(
    source_document: KnowledgeDocument,
    chunk: KnowledgeDocument,
    *,
    strategy_id: str,
) -> KnowledgeDocument:
    validated = KnowledgeDocument.model_validate(chunk.model_dump(mode="python"))

    if validated.identity.parent_document_id != source_document.identity.document_id:
        raise ValueError("Chunk parent_document_id must match the source document document_id")

    if validated.identity.root_document_id != source_document.identity.root_document_id:
        raise ValueError(
            "Chunk root_document_id must match the source document root_document_id"
        )

    if validated.scope != source_document.scope:
        raise ValueError("Chunk scope must match the source document scope")

    source_provenance = source_document.provenance
    chunk_provenance = validated.provenance
    if (
        chunk_provenance.source_kind != source_provenance.source_kind
        or chunk_provenance.source_id != source_provenance.source_id
        or chunk_provenance.source_parent_id != source_provenance.source_parent_id
        or chunk_provenance.provider_id != source_provenance.provider_id
        or chunk_provenance.source_revision != source_provenance.source_revision
        or chunk_provenance.source_uri != source_provenance.source_uri
    ):
        raise ValueError("Chunk provenance source fields must match the source document")

    chunk_index = validated.metadata.get(ChunkMetadataKey.CHUNK_INDEX.value)
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("Chunk metadata chunk_index must be a non-negative int")

    chunk_id = validated.metadata.get(ChunkMetadataKey.CHUNK_ID.value)
    if chunk_id != validated.identity.document_id:
        raise ValueError("Chunk metadata chunk_id must match identity.document_id")

    chunk_strategy = validated.metadata.get(ChunkMetadataKey.CHUNK_STRATEGY.value)
    if chunk_strategy != strategy_id:
        raise ValueError("Chunk metadata chunk_strategy must match the requested strategy_id")

    chunk_size = validated.metadata.get(ChunkMetadataKey.CHUNK_SIZE.value)
    if chunk_size != len(validated.content):
        raise ValueError("Chunk metadata chunk_size must match len(content)")

    expected_content_hash = _sha256_hex(validated.content)
    if validated.provenance.content_hash != expected_content_hash:
        raise ValueError("Chunk provenance content_hash must match SHA-256 of content")

    expected_document_id = _derive_chunk_document_id(
        source_document_id=source_document.identity.document_id,
        strategy_id=strategy_id,
        chunk_index=chunk_index,
        content=validated.content,
    )
    if validated.identity.document_id != expected_document_id:
        raise ValueError("Chunk identity.document_id must match the deterministic derived id")

    return validated


def build_derived_chunk(
    source_document: KnowledgeDocument,
    *,
    content: str,
    strategy_id: str,
    chunk_index: int,
    metadata_updates: Mapping[str, JsonValue] | None = None,
) -> KnowledgeDocument:
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise ValueError("strategy_id must be a non-empty string")

    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative int")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("chunk content must be a non-empty string")

    if metadata_updates:
        forbidden = _CANONICAL_CHUNK_METADATA_KEYS.intersection(metadata_updates)
        if forbidden:
            raise ValueError(
                "metadata_updates must not override canonical chunk metadata keys: "
                f"{sorted(forbidden)}"
            )

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

    chunk = KnowledgeDocument(
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

    return validate_derived_chunk(source_document, chunk, strategy_id=strategy_id)
