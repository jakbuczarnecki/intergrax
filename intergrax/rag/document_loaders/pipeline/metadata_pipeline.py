# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, List

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    copy_parser_runtime_state,
)
from intergrax.rag.document_loaders.contracts.metadata_provider import BaseMetadataProvider


class MetadataPipeline:
    """
    Deterministic pipeline executing metadata providers in sequence.
    """

    def __init__(
        self,
        providers: Iterable[BaseMetadataProvider],
    ) -> None:
        self._providers: List[BaseMetadataProvider] = list(providers)

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:

        docs: List[KnowledgeDocument] = list(documents)

        for provider in self._providers:
            result = list(provider.enrich(docs, source))

            if len(result) != len(docs):
                raise ValueError(
                    f"metadata provider {type(provider).__name__} changed document count: "
                    f"expected {len(docs)}, got {len(result)}"
                )

            validated: List[KnowledgeDocument] = []
            for source_doc, result_doc in zip(docs, result, strict=True):
                if not isinstance(result_doc, KnowledgeDocument):
                    raise ValueError(
                        f"metadata provider {type(provider).__name__} returned "
                        f"{type(result_doc).__name__}, expected KnowledgeDocument"
                    )

                validated_result = KnowledgeDocument.model_validate(
                    result_doc.model_dump(mode="python")
                )

                if validated_result.schema_version != source_doc.schema_version:
                    raise ValueError("metadata provider changed schema_version")
                if validated_result.identity != source_doc.identity:
                    raise ValueError("metadata provider changed identity")
                if validated_result.scope != source_doc.scope:
                    raise ValueError("metadata provider changed scope")
                if validated_result.content != source_doc.content:
                    raise ValueError("metadata provider changed content")
                if validated_result.provenance != source_doc.provenance:
                    raise ValueError("metadata provider changed provenance")

                validated.append(
                    copy_parser_runtime_state(source_doc, validated_result)
                )

            docs = validated

        return docs
