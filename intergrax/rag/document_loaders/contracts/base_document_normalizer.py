# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from pathlib import Path

from intergrax.knowledge.contracts import KnowledgeDocument


class BaseDocumentNormalizer(ABC):
    """
    Contract for document normalization step.

    Normalizers are responsible for stabilizing document content
    before metadata enrichment and chunking.

    Responsibilities may include:
    - whitespace normalization
    - newline normalization
    - OCR artifact cleanup
    - header/footer removal
    """

    @abstractmethod
    def normalize(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        """
        Normalize document content.

        Implementations must:
        - preserve document count
        - preserve metadata
        - avoid semantic changes to text
        """
        raise NotImplementedError
