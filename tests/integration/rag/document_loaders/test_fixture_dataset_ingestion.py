# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pathlib import Path
import pytest

from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_documents_loader


pytestmark = pytest.mark.integration


def test_docx_ingestion():

    DATASET = Path("tests/fixtures/documents/docx")

    loader = create_default_documents_loader()

    docs = loader.load_documents(str(DATASET))

    assert docs is not None
    assert len(docs) > 0

    for doc in docs:
        assert doc.page_content
        assert "source" in doc.metadata


def test_html_ingestion():

    DATASET = Path("tests/fixtures/documents/html")

    loader = create_default_documents_loader()

    docs = loader.load_documents(str(DATASET))

    assert docs
    assert len(docs) > 0

    for doc in docs:
        assert doc.page_content
        assert "source" in doc.metadata


def test_xlsx_ingestion():

    DATASET = Path("tests/fixtures/documents/xlsx")

    loader = create_default_documents_loader()

    docs = loader.load_documents(str(DATASET))

    assert docs
    assert len(docs) > 0

    for doc in docs:
        assert doc.page_content
        assert "source" in doc.metadata


def test_txt_ingestion():

    DATASET = Path("tests/fixtures/documents/txt")

    loader = create_default_documents_loader()

    docs = loader.load_documents(str(DATASET))

    assert docs
    assert len(docs) > 0

    for doc in docs:
        assert doc.page_content
        assert "source" in doc.metadata