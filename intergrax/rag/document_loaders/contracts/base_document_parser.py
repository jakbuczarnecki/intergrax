# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment


class BaseDocumentParser(ABC):
    """
    Contract for pluggable document parsing backends used INSIDE a format handler.

    Examples:
    - docling backend
    - unstructured backend (generic, not format-specific handler)
    - custom enterprise parser

    This contract is NOT a replacement for BaseDocumentHandler.
    BaseDocumentHandler remains responsible for:
    - supports(source)
    - confidence(source)
    - public load(source) boundary used by the registry

    BaseDocumentParser is an internal delegation mechanism for handler.load(...),
    enabling deterministic backend selection (prefer/fallback) without introducing
    separate handlers per external tool.

    Parser extracts provider-neutral fragments; handler/loader owns mapping into
    scoped KnowledgeDocument; no tenant or identity fallback at parser stage.
    """

    @classmethod
    @abstractmethod
    def parser_id(cls) -> str:
        """
        Stable identifier of the parser backend implementation.
        Must be deterministic and constant (e.g. "docling", "unstructured", "custom.acme.v1").
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if this backend can be used in the current environment
        (dependencies installed, binaries present, configured, etc.).
        Must be deterministic (no network calls).
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        """
        Parse the source into provider-neutral document fragments.

        Parameters
        ----------
        source : str
            Source URI (file path, HTTP URL, S3 URI, etc.).

        Returns
        -------
        Sequence[ParsedDocumentFragment]
        """
        raise NotImplementedError
