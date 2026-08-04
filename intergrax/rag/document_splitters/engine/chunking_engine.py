# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.registry.strategy_registry import ChunkingStrategyRegistry


class ChunkingEngine:
    """
    Execution engine responsible for applying a chunking strategy
    to a collection of documents.

    The engine resolves the strategy from the registry and delegates
    the chunking execution to the selected strategy.

    A production fail-safe is implemented to guarantee that documents
    are never silently dropped from the ingestion pipeline.
    """

    def __init__(
        self,
        registry: ChunkingStrategyRegistry,
    ) -> None:
        self._registry = registry

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str,
    ) -> Sequence[KnowledgeDocument]:
        """
        Execute chunking using the specified strategy.

        Parameters
        ----------
        documents : Sequence[KnowledgeDocument]
            Documents produced by the ingestion pipeline.

        strategy_id : str
            Identifier of the chunking strategy.

        Returns
        -------
        Sequence[KnowledgeDocument]
            Chunked documents.
        """

        strategy: BaseChunkingStrategy = self._registry.resolve(strategy_id)

        raw_chunks = list(strategy.chunk(documents))

        input_by_id = {document.identity.document_id: document for document in documents}
        validated_by_parent: dict[str, list[KnowledgeDocument]] = defaultdict(list)
        seen_ids: set[str] = set()

        for chunk in raw_chunks:
            if not isinstance(chunk, KnowledgeDocument):
                raise TypeError(
                    f"Chunking strategy {strategy_id!r} returned non-KnowledgeDocument: "
                    f"{type(chunk)!r}"
                )

            revalidated = KnowledgeDocument.model_validate(chunk.model_dump(mode="python"))
            chunk_id = revalidated.identity.document_id
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk document_id: {chunk_id}")
            seen_ids.add(chunk_id)

            parent_id = revalidated.identity.parent_document_id
            if parent_id is None or parent_id not in input_by_id:
                raise ValueError(
                    f"Chunk parent_document_id {parent_id!r} does not match an input document"
                )

            source = input_by_id[parent_id]
            self._assert_chunk_lineage(source, revalidated)
            validated_by_parent[parent_id].append(revalidated)

        result: list[KnowledgeDocument] = []
        for source in documents:
            source_id = source.identity.document_id
            source_chunks = validated_by_parent.get(source_id, [])
            if source_chunks:
                result.extend(source_chunks)
                continue

            result.append(
                build_derived_chunk(
                    source,
                    content=source.content,
                    strategy_id=strategy_id,
                    chunk_index=0,
                    metadata_updates={"chunk_fallback": True},
                )
            )

        return result

    @staticmethod
    def _assert_chunk_lineage(
        source: KnowledgeDocument,
        chunk: KnowledgeDocument,
    ) -> None:
        if chunk.identity.root_document_id != source.identity.root_document_id:
            raise ValueError(
                "Chunk root_document_id must match the source document root_document_id"
            )

        if chunk.scope != source.scope:
            raise ValueError("Chunk scope must match the source document scope")

        source_provenance = source.provenance
        chunk_provenance = chunk.provenance
        if (
            chunk_provenance.source_kind != source_provenance.source_kind
            or chunk_provenance.source_id != source_provenance.source_id
            or chunk_provenance.source_parent_id != source_provenance.source_parent_id
            or chunk_provenance.provider_id != source_provenance.provider_id
            or chunk_provenance.source_revision != source_provenance.source_revision
            or chunk_provenance.source_uri != source_provenance.source_uri
        ):
            raise ValueError("Chunk provenance source fields must match the source document")
