# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import re
from typing import Sequence
from pathlib import Path

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.contracts.base_document_normalizer import (
    BaseDocumentNormalizer,
)


class WhitespaceNormalizer(BaseDocumentNormalizer):
    """
    Normalizes whitespace artifacts in extracted document text.
    """

    _multi_space_re = re.compile(r"[ \t]+")
    _multi_newline_re = re.compile(r"\n{3,}")

    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:

        normalized: list[KnowledgeDocument] = []

        for doc in documents:
            text = doc.content or ""

            text = self._multi_space_re.sub(" ", text)
            text = self._multi_newline_re.sub("\n\n", text)
            text = text.strip()

            payload = doc.model_dump(mode="python")
            payload["content"] = text
            normalized.append(KnowledgeDocument.model_validate(payload))

        return normalized
