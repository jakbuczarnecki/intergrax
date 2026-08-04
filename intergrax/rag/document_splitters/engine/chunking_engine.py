# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import (
    build_derived_chunk,
    validate_derived_chunk,
)
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

        source_documents = list(documents)
        input_by_id: dict[str, KnowledgeDocument] = {}
        for document in source_documents:
            if not isinstance(document, KnowledgeDocument):
                raise TypeError(
                    f"Chunking input must be KnowledgeDocument instances, got {type(document)!r}"
                )
            revalidated_source = KnowledgeDocument.model_validate(
                document.model_dump(mode="python")
            )
            source_id = revalidated_source.identity.document_id
            if source_id in input_by_id:
                raise ValueError(f"Duplicate source document_id: {source_id}")
            input_by_id[source_id] = revalidated_source

        revalidated_sources = [
            input_by_id[document.identity.document_id] for document in source_documents
        ]

        raw_chunks = list(strategy.chunk(revalidated_sources))

        validated_by_parent: dict[str, list[KnowledgeDocument]] = defaultdict(list)
        seen_ids: set[str] = set()

        for chunk in raw_chunks:
            if not isinstance(chunk, KnowledgeDocument):
                raise TypeError(
                    f"Chunking strategy {strategy_id!r} returned non-KnowledgeDocument: "
                    f"{type(chunk)!r}"
                )

            revalidated = KnowledgeDocument.model_validate(chunk.model_dump(mode="python"))

            parent_id = revalidated.identity.parent_document_id
            if parent_id is None or parent_id not in input_by_id:
                raise ValueError(
                    f"Chunk parent_document_id {parent_id!r} does not match an input document"
                )

            source = input_by_id[parent_id]
            validated = validate_derived_chunk(
                source, revalidated, strategy_id=strategy_id
            )
            chunk_id = validated.identity.document_id
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk document_id: {chunk_id}")
            seen_ids.add(chunk_id)
            validated_by_parent[parent_id].append(validated)

        for source in revalidated_sources:
            source_id = source.identity.document_id
            if validated_by_parent.get(source_id):
                continue

            fallback = build_derived_chunk(
                source,
                content=source.content,
                strategy_id=strategy_id,
                chunk_index=0,
                metadata_updates={"chunk_fallback": True},
            )
            validated = validate_derived_chunk(
                source, fallback, strategy_id=strategy_id
            )
            chunk_id = validated.identity.document_id
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk document_id: {chunk_id}")
            seen_ids.add(chunk_id)
            validated_by_parent[source_id].append(validated)

        result: list[KnowledgeDocument] = []
        for source in revalidated_sources:
            result.extend(validated_by_parent[source.identity.document_id])

        result_ids = [chunk.identity.document_id for chunk in result]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("Chunking result contains duplicate document_id values")

        return result
