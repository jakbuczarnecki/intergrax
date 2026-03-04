# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence, Literal

import docx
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader

from intergrax.rag.document_loaders.config.document_loader_config import (
    DEFAULT_BUILTIN_HANDLER_CONFIDENCE,
)
from intergrax.rag.document_loaders.contracts.base_document_handler import (
    BaseDocumentHandler,
)
from intergrax.rag.document_loaders.contracts.base_document_parser import (
    BaseDocumentParser,
)

EXTRACTION_STRATEGY = Literal["auto", "fulltext", "paragraphs", "headings"]


class DocSmartDocumentHandler(BaseDocumentHandler):

    def __init__(self, extraction_strategy: EXTRACTION_STRATEGY = "auto") -> None:
        self._extraction_strategy = extraction_strategy

    def supports(self, source: str) -> bool:
        s = source.lower()
        return s.endswith(".docx") or s.endswith(".doc")

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def build_parsers(self) -> List[BaseDocumentParser]:
        return [
            DocSmartParser(strategy=self._extraction_strategy)
        ]


class DocSmartParser(BaseDocumentParser):

    def __init__(self, strategy: EXTRACTION_STRATEGY):
        self._strategy = strategy

    @classmethod
    def parser_id(cls) -> str:
        return "doc_smart"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        strategy = self._strategy

        if strategy == "auto":
            strategy = "fulltext"

        if strategy == "fulltext":
            loader = Docx2txtLoader(source)
            return loader.load()

        if strategy in ("paragraphs", "headings"):
            loader = DocxParagraphLoader(source, mode=strategy)
            return loader.load()

        raise RuntimeError("Invalid extraction strategy state")


class DocxParagraphLoader:

    def __init__(self, path: str, mode: EXTRACTION_STRATEGY = "paragraphs"):
        if docx is None:
            raise ImportError(
                "python-docx is required for DocxParagraphLoader (pip install python-docx)"
            )
        self.path = path
        self.mode = mode

    def _is_heading(self, para) -> tuple[bool, int]:

        style = getattr(para.style, "name", "") or ""
        if not style:
            return (False, 0)

        s = style.lower()

        if s.startswith("heading"):
            for i in range(1, 10):
                if s == f"heading {i}":
                    return (True, i)
            return (True, 1)

        return (False, 0)

    def load(self):

        d = docx.Document(self.path)

        items = []
        heading_stack: list[str] = []
        section_ix = 0
        para_ix = 0

        for p in d.paragraphs:

            text = (p.text or "").strip()

            if not text:
                continue

            is_head, level = self._is_heading(p)

            if is_head:

                heading_stack = heading_stack[:max(level - 1, 0)]
                heading_stack.append(text)

                section_ix += 1

                if self.mode == "headings":

                    meta = {
                        "doc_type": "docx",
                        "source_path": self.path,
                        "source_name": str(self.path).split("/")[-1],
                        "section_ix": section_ix,
                        "heading_path": " / ".join(heading_stack),
                        "is_heading": True,
                    }

                    items.append(Document(page_content=text, metadata=meta))

                continue

            if self.mode == "paragraphs":

                para_ix += 1

                meta = {
                    "doc_type": "docx",
                    "source_path": self.path,
                    "source_name": str(self.path).split("/")[-1],
                    "section_ix": section_ix,
                    "para_ix": para_ix,
                    "heading_path": " / ".join(heading_stack),
                    "is_heading": False,
                }

                items.append(Document(page_content=text, metadata=meta))

        return items