# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser


class BaseDocumentHandler(ABC):
    """
    Contract for document format handlers used in the Intergrax RAG ingestion system.

    Handlers convert a source URI into a sequence of KnowledgeDocument objects.
    """

    @abstractmethod
    def supports(self, source: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def confidence(self, source: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def build_parsers(self) -> List[BaseDocumentParser]:
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        source: str,
        *,
        scope: KnowledgeDocumentScope,
    ) -> Sequence[KnowledgeDocument]:
        raise NotImplementedError
