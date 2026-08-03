# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser


pytestmark = pytest.mark.unit


class _FakeBackend:
    def parser_id(self) -> str:
        return "fake.backend"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source: str) -> list[ParsedDocumentFragment]:
        return [
            ParsedDocumentFragment(
                text="backend text",
                metadata={"page": 1},
                native_handle={"native": True},
            )
        ]


def test_catalog_parser_returns_parsed_document_fragment() -> None:
    parser = CatalogDocumentParser(_FakeBackend())

    fragments = parser.load("doc.pdf")

    assert len(fragments) == 1
    assert isinstance(fragments[0], ParsedDocumentFragment)
    assert fragments[0].text == "backend text"


def test_catalog_parser_preserves_backend_metadata_and_adds_loader_metadata() -> None:
    parser = CatalogDocumentParser(_FakeBackend())

    fragments = parser.load("doc.pdf")
    metadata = fragments[0].metadata

    assert metadata["page"] == 1
    assert metadata["parser"] == "fake.backend"
    assert metadata["source"] == "doc.pdf"
    assert metadata["position"] == 0
    assert "document_id" in metadata


def test_catalog_parser_keeps_native_handle_in_typed_field() -> None:
    backend_fragment = _FakeBackend().parse_file("doc.pdf")[0]
    parser = CatalogDocumentParser(_FakeBackend())

    fragments = parser.load("doc.pdf")

    assert fragments[0].native_handle == {"native": True}
    assert backend_fragment.native_handle == {"native": True}
    assert "_docling_document" not in fragments[0].metadata


def test_catalog_parser_does_not_mutate_backend_fragment() -> None:
    backend = _FakeBackend()
    original = backend.parse_file("doc.pdf")[0]
    original_metadata = dict(original.metadata)

    CatalogDocumentParser(backend).load("doc.pdf")

    assert original.metadata == original_metadata
    assert original.text == "backend text"
