# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument


class BaseMetadataProvider(ABC):
    """
    Contract for metadata enrichment used in the Intergrax RAG document loading pipeline.

    Implementations may add or modify metadata fields on KnowledgeDocument objects.
    """

    @abstractmethod
    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        source: Path | str,
    ) -> Sequence[KnowledgeDocument]:
        """
        Enrich documents with metadata.

        Parameters
        ----------
        documents : Sequence[KnowledgeDocument]
            Documents returned by a loader provider.

        source : Path | str
            Original source location.

        Returns
        -------
        Sequence[KnowledgeDocument]
            Documents with additional metadata.
        """
        raise NotImplementedError
