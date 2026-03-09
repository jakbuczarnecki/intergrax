
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from langchain_core.documents import Document


class BaseDocumentsSplitter(ABC):
    
    @abstractmethod
    def split_documents(
        self,
        documents: Sequence[Document],
        strategy_id: Optional[str] = None,
    ) -> Sequence[Document]:
        """
        Split documents using a configured chunking strategy.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents produced by the ingestion pipeline.

        strategy_id : str
            Identifier of the chunking strategy.

        Returns
        -------
        Sequence[Document]
            Chunked documents produced by the selected strategy.
        """

        raise NotImplementedError