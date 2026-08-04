# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Iterable, Sequence, List
from pathlib import Path

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    copy_parser_runtime_state,
)
from intergrax.rag.document_loaders.contracts.base_document_normalizer import (
    BaseDocumentNormalizer,
)


class NormalizerPipeline:
    """
    Deterministic pipeline executing document normalizers in sequence.
    """

    def __init__(
        self,
        normalizers: Iterable[BaseDocumentNormalizer],
    ) -> None:

        self._normalizers: List[BaseDocumentNormalizer] = list(normalizers)

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:

        docs: List[KnowledgeDocument] = list(documents)

        for normalizer in self._normalizers:
            result = list(normalizer.normalize(docs, source))

            if len(result) != len(docs):
                raise ValueError(
                    f"normalizer {type(normalizer).__name__} changed document count: "
                    f"expected {len(docs)}, got {len(result)}"
                )

            validated: List[KnowledgeDocument] = []
            for source_doc, result_doc in zip(docs, result, strict=True):
                if not isinstance(result_doc, KnowledgeDocument):
                    raise ValueError(
                        f"normalizer {type(normalizer).__name__} returned "
                        f"{type(result_doc).__name__}, expected KnowledgeDocument"
                    )

                validated_result = KnowledgeDocument.model_validate(
                    result_doc.model_dump(mode="python")
                )

                if validated_result.schema_version != source_doc.schema_version:
                    raise ValueError("normalizer changed schema_version")
                if validated_result.identity != source_doc.identity:
                    raise ValueError("normalizer changed identity")
                if validated_result.scope != source_doc.scope:
                    raise ValueError("normalizer changed scope")
                if validated_result.metadata != source_doc.metadata:
                    raise ValueError("normalizer changed metadata")
                if validated_result.provenance != source_doc.provenance:
                    raise ValueError("normalizer changed provenance")

                validated.append(
                    copy_parser_runtime_state(source_doc, validated_result)
                )

            docs = validated

        return docs
