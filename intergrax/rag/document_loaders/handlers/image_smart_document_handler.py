# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List


from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.config.document_loader_config import (
    DEFAULT_BUILTIN_HANDLER_CONFIDENCE,
)
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.parsers.image_smart_parser import ImageSmartParser


class ImageSmartDocumentHandler(BaseDocumentHandler):

    def __init__(
        self,
        *,
        ocr_lang: str = "eng",
        ocr_psm: int | None = None,
        ocr_oem: int | None = None,
        extract_exif: bool = True,
        max_image_dim: int | None = None,
        text_mode: str = "both",
        caption_llm=None,
        both_joiner: str = "\n\n---\n\n",
    ) -> None:
        self._ocr_lang = ocr_lang
        self._ocr_psm = ocr_psm
        self._ocr_oem = ocr_oem
        self._extract_exif = extract_exif
        self._max_image_dim = max_image_dim
        self._text_mode = text_mode
        self._caption_llm = caption_llm
        self._both_joiner = both_joiner

    def supports(self, source: str) -> bool:
        s = source.lower()
        return (
            s.endswith(".jpg")
            or s.endswith(".jpeg")
            or s.endswith(".png")
            or s.endswith(".tiff")
            or s.endswith(".bmp")
            or s.endswith(".webp")
            or s.endswith(".heic")
            or s.endswith(".heif")
        )

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def build_parsers(self) -> List[BaseDocumentParser]:

        return [
            ImageSmartParser(
                ocr_lang=self._ocr_lang,
                ocr_psm=self._ocr_psm,
                ocr_oem=self._ocr_oem,
                extract_exif=self._extract_exif,
                max_image_dim=self._max_image_dim,
                text_mode=self._text_mode,
                caption_llm=self._caption_llm,
                both_joiner=self._both_joiner,
            )
        ]