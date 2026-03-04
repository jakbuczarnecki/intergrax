# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document


class BaseMetadataProvider(ABC):
    """
    Contract for metadata enrichment used in the Intergrax RAG document loading pipeline.

    Implementations may add or modify metadata fields on Document objects.
    """

    @abstractmethod
    def enrich(
        self,
        documents: Sequence[Document],
        source: Path | str,
    ) -> Sequence[Document]:
        """
        Enrich documents with metadata.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents returned by a loader provider.

        source : Path | str
            Original source location.

        Returns
        -------
        Sequence[Document]
            Documents with additional metadata.
        """
        raise NotImplementedError