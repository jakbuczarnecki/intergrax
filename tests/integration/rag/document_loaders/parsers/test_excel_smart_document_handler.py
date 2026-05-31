# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.excel_smart_document_handler import (
    ExcelSmartDocumentHandler,
)

pytestmark = pytest.mark.integration


def _create_csv(path: Path) -> None:

    df = pd.DataFrame(
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
    )

    df.to_csv(path, index=False)


def test_excel_handler_supports_extensions():

    handler = ExcelSmartDocumentHandler()

    assert handler.supports("file.xlsx") is True
    assert handler.supports("file.xls") is True
    assert handler.supports("file.csv") is True
    assert handler.supports("file.tsv") is True
    assert handler.supports("file.txt") is False


def test_excel_handler_confidence():

    handler = ExcelSmartDocumentHandler()

    assert handler.confidence("file.xlsx") == GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence


def test_excel_handler_builds_parser():

    handler = ExcelSmartDocumentHandler()

    parsers = handler.build_parsers()

    assert len(parsers) >= 1
    assert parsers[0].parser_id() == "openpyxl"


def test_excel_handler_loads_csv(tmp_path: Path):

    csv_path = tmp_path / "sample.csv"

    _create_csv(csv_path)

    handler = ExcelSmartDocumentHandler()

    docs = handler.load(str(csv_path))

    assert docs
    assert all(isinstance(d, Document) for d in docs)

    content = " ".join(d.page_content for d in docs)

    assert "Alice" in content
    assert "Bob" in content