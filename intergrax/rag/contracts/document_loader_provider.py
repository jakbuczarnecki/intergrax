# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document


class BaseDocumentLoaderProvider(ABC):
    """
    Contract for document loader providers used by the RAG ingestion system.

    Implementations are responsible for extracting Document objects
    from a supported source (file, URL, etc.).
    """

    @abstractmethod
    def supports(self, source: Path | str) -> bool:
        """
        Determine whether this provider can load the given source.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, source: Path | str) -> Sequence[Document]:
        """
        Extract documents from the given source.
        """
        raise NotImplementedError