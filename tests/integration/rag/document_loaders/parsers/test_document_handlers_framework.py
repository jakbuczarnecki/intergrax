# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import sys

import pytest
from pathlib import Path

from langchain_core.documents import Document

from intergrax.rag.document_loaders.handlers.pdf_smart_document_handler import (
    PdfSmartDocumentHandler,
)


pytestmark = pytest.mark.integration

def _create_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Intergrax handler test")
    doc.save(path)
    doc.close()


@pytest.mark.parametrize(
    "handler_cls,extension,creator",
    [
        (PdfSmartDocumentHandler, ".pdf", _create_pdf),
    ],
)
def test_document_handler_contract(tmp_path: Path, handler_cls, extension, creator):

    file_path = tmp_path / f"sample{extension}"

    creator(file_path)

    handler = handler_cls()

    # supports
    assert handler.supports(str(file_path)) is True

    # parser construction
    parsers = handler.build_parsers()

    assert parsers
    assert hasattr(parsers[0], "load")

    # real parsing
    docs = handler.load(str(file_path))

    assert docs
    assert all(isinstance(d, Document) for d in docs)