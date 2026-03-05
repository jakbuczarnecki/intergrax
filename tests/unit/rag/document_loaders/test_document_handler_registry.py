# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.rag.document_loaders.registry.document_handler_registry import (
    DocumentHandlerRegistry,
)
from intergrax.rag.document_loaders.contracts.base_document_handler import (
    BaseDocumentHandler,
)

pytestmark = pytest.mark.unit


class _DummyHandler(BaseDocumentHandler):
    """
    Minimal deterministic test handler implementing BaseDocumentHandler contract.

    NOTE:
    Registry tests only use supports() and confidence().
    build_parsers() is implemented only to satisfy the ABC contract.
    """

    def __init__(self, supported_suffix: str, score: float) -> None:
        self.supported_suffix = supported_suffix
        self.score = score

    def supports(self, source: str) -> bool:
        return source.endswith(self.supported_suffix)

    def confidence(self, source: str) -> float:
        return self.score

    def build_parsers(self):
        return []


def test_registry_resolves_single_handler() -> None:
    registry = DocumentHandlerRegistry()

    handler = _DummyHandler(".pdf", 0.9)
    registry.register(handler)

    resolved = registry.resolve("file.pdf")

    assert resolved is handler


def test_registry_selects_highest_confidence() -> None:
    registry = DocumentHandlerRegistry()

    handler_low = _DummyHandler(".pdf", 0.5)
    handler_high = _DummyHandler(".pdf", 0.9)

    registry.register(handler_low)
    registry.register(handler_high)

    resolved = registry.resolve("file.pdf")

    assert resolved is handler_high


def test_registry_filters_by_supports() -> None:
    registry = DocumentHandlerRegistry()

    handler_pdf = _DummyHandler(".pdf", 0.8)
    handler_doc = _DummyHandler(".docx", 0.9)

    registry.register(handler_pdf)
    registry.register(handler_doc)

    resolved = registry.resolve("file.pdf")

    assert resolved is handler_pdf


def test_registry_raises_when_no_handler_supports_source() -> None:
    registry = DocumentHandlerRegistry()

    handler = _DummyHandler(".pdf", 0.8)
    registry.register(handler)

    with pytest.raises(RuntimeError):
        registry.resolve("file.txt")