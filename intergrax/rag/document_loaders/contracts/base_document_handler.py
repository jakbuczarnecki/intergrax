# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document


class BaseDocumentHandler(ABC):
    """
    Contract for document format handlers used in the Intergrax RAG ingestion system.

    Handlers are responsible for converting a source file into a sequence
    of LangChain Document objects.

    The handler does NOT perform orchestration, metadata enrichment,
    chunking, or indexing.
    """

    @abstractmethod
    def supports(self, source: Path) -> bool:
        """
        Determine whether this handler supports the given file.

        Parameters
        ----------
        source : Path
            Path to the source document.

        Returns
        -------
        bool
            True if this handler can process the file.
        """
        raise NotImplementedError

    @abstractmethod
    def confidence(self, source: Path) -> float:
        """
        Estimate how well this handler can process the document.

        Returns
        -------
        float
            Value in range [0.0, 1.0].

            0.0  → handler should not be used
            1.0  → handler is ideal for this document
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, source: Path) -> Sequence[Document]:
        """
        Load a document and convert it into LangChain Document objects.

        Parameters
        ----------
        source : Path
            Path to the source document.

        Returns
        -------
        Sequence[Document]
            Extracted documents.
        """
        raise NotImplementedError