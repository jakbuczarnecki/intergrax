# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.pdf_smart_document_handler import (
    PdfSmartDocumentHandler,
)

pytestmark = pytest.mark.integration


def _create_minimal_pdf(path: Path) -> None:
    """
    Create a minimal valid PDF for testing using PyMuPDF.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello Intergrax PDF test")
    doc.save(path)
    doc.close()


def test_pdf_handler_supports_extension():

    handler = PdfSmartDocumentHandler()

    assert handler.supports("file.pdf") is True
    assert handler.supports("file.txt") is False


def test_pdf_handler_confidence():

    handler = PdfSmartDocumentHandler()

    assert handler.confidence("file.pdf") == GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence


def test_pdf_handler_builds_parser():

    handler = PdfSmartDocumentHandler()

    parsers = handler.build_parsers()

    assert len(parsers) == 1
    assert parsers[0].parser_id() == "pymupdf"


def test_pdf_handler_loads_pdf(tmp_path: Path):

    pdf_path = tmp_path / "sample.pdf"
    _create_minimal_pdf(pdf_path)

    handler = PdfSmartDocumentHandler(enable_ocr=False)

    docs = handler.load(str(pdf_path))

    assert docs
    assert any("Hello Intergrax PDF test" in d.page_content for d in docs)