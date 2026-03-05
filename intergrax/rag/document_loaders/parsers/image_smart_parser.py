# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from intergrax.multimedia.image_smart_loader import ImageSmartLoader
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata

class ImageSmartParser(BaseDocumentParser):

    def __init__(
        self,
        *,
        ocr_lang: str,
        ocr_psm: int | None,
        ocr_oem: int | None,
        extract_exif: bool,
        max_image_dim: int | None,
        text_mode: str,
        caption_llm,
        both_joiner: str,
    ):

        self._ocr_lang = ocr_lang
        self._ocr_psm = ocr_psm
        self._ocr_oem = ocr_oem
        self._extract_exif = extract_exif
        self._max_image_dim = max_image_dim
        self._text_mode = text_mode
        self._caption_llm = caption_llm
        self._both_joiner = both_joiner

    @classmethod
    def parser_id(cls) -> str:
        return "image_smart"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        loader = ImageSmartLoader(
            source,
            ocr_lang=self._ocr_lang,
            ocr_psm=self._ocr_psm,
            ocr_oem=self._ocr_oem,
            extract_exif=self._extract_exif,
            max_image_dim=self._max_image_dim,
            text_mode=self._text_mode,
            caption_llm=self._caption_llm,
            both_joiner=self._both_joiner,
        )

        docs = loader.load()

        result: list[Document] = []

        for i, d in enumerate(docs):

            metadata = build_loader_metadata(
                source=source,
                parser=self.parser_id(),
                position=i,
            )

            metadata.update(d.metadata or {})

            result.append(
                Document(
                    page_content=d.page_content,
                    metadata=metadata,
                )
            )

        return result