# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """
    Contract for document loader providers used by the Intergrax RAG infrastructure.

    Implementations are responsible for loading raw files and converting them
    into normalized `Document` objects used in downstream RAG pipelines.
    """

    @abstractmethod
    def load(self, source: Path | str) -> Sequence[Document]:
        """
        Load documents from a given source.

        Parameters
        ----------
        source : Path | str
            Filesystem path, directory, or resource identifier.

        Returns
        -------
        Sequence[Document]
            Normalized document objects ready for downstream processing.
        """
        raise NotImplementedError

    @abstractmethod
    def supports(self, source: Path | str) -> bool:
        """
        Determine whether this loader can handle the provided source.

        Parameters
        ----------
        source : Path | str

        Returns
        -------
        bool
        """
        raise NotImplementedError