# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.tools.providers.document.contracts import DocumentParsePreviewInput
from intergrax.tools.providers.document.service import document_parse_preview
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeDocumentParser(DocumentParser):
    def parser_id(self) -> str:
        return "fake-parser"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source_path: str) -> list[ParsedDocumentFragment]:
        return [
            ParsedDocumentFragment(text=f"chunk-{index}", metadata={"index": index})
            for index in range(10)
        ]


def test_document_parse_preview_truncates_fragments() -> None:
    ctx = ToolWiringContext(document_parser=FakeDocumentParser())
    out = document_parse_preview(
        ctx,
        DocumentParsePreviewInput(source_path="sample.pdf", max_fragments=3),
    )
    assert out.parser_id == "fake-parser"
    assert out.fragment_count == 3
    assert out.truncated is True
