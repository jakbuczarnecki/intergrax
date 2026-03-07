# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from langchain_core.documents import Document


class BaseChunkingStrategy(ABC):
    """
    Contract for document chunking strategies used in the Intergrax RAG pipeline.

    Chunking strategies transform input documents into chunked documents suitable
    for embedding and retrieval systems.

    Implementations may use different algorithms such as:
    - token-based chunking
    - recursive chunking
    - semantic chunking
    - structure-aware chunking
    - agentic chunking

    All strategies must produce deterministic results for the same input.
    """

    @classmethod
    @abstractmethod
    def strategy_id(cls) -> str:
        """
        Stable identifier of the chunking strategy.

        Examples
        --------
        "recursive"
        "semantic"
        "docling"
        "parent_child"
        "agentic"
        """
        raise NotImplementedError

    @abstractmethod
    def chunk(
        self,
        documents: Sequence[Document],
    ) -> Sequence[Document]:
        """
        Chunk input documents.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents produced by the ingestion pipeline.

        Returns
        -------
        Sequence[Document]
            Chunked documents.

        Requirements
        ------------
        Implementations must:
        - preserve original metadata fields
        - preserve source and document identifiers
        - maintain deterministic ordering of chunks
        - avoid semantic modification of the original text
        """
        raise NotImplementedError