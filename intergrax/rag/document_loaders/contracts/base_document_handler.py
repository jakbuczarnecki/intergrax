# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.pipeline.parser_pipeline import ParserPipeline


class BaseDocumentHandler(ABC):
    """
    Contract for document format handlers used in the Intergrax RAG ingestion system.

    Handlers convert a source URI into a sequence of LangChain Document objects.
    """

    @abstractmethod
    def supports(self, source: str) -> bool:
        """
        Determine whether this handler supports the given source.

        Parameters
        ----------
        source : str
            Source URI (file path, HTTP URL, S3 URI, etc.).

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @abstractmethod
    def confidence(self, source: str) -> float:
        """
        Estimate how well this handler can process the source.

        Returns
        -------
        float
            Value in range [0.0, 1.0].
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, source: str) -> Sequence[Document]:
        """
        Load documents from the given source.

        Parameters
        ----------
        source : str
            Source URI.

        Returns
        -------
        Sequence[Document]
        """
        raise NotImplementedError


    @abstractmethod
    def build_parsers(self) -> List[BaseDocumentParser]:
        """
        Return ordered list of parsers.

        Order defines priority.
        """
        raise NotImplementedError


    def load(self, source: str) -> Sequence[Document]:
        """
        Execute parser pipeline.
        """
        parsers = self.build_parsers()

        pipeline = ParserPipeline(parsers)

        return pipeline.parse(source)