# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Iterable, Sequence, List
from pathlib import Path

from langchain_core.documents import Document

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
        documents: Sequence[Document],
        source: Path | str,
    ) -> Sequence[Document]:

        docs: Sequence[Document] = documents

        for normalizer in self._normalizers:
            docs = normalizer.normalize(docs, source)

        return docs