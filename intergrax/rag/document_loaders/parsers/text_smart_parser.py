# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from codecs import BOM_UTF8
from pathlib import Path
from typing import Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata


def _read_text(source: str) -> str:
    data = Path(source).read_bytes()

    if data.startswith(BOM_UTF8):
        return data.decode("utf-8-sig")

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as utf8_error:
        from chardet import detect

        detected = detect(data)
        encoding = detected.get("encoding")
        confidence = detected.get("confidence") or 0.0

        if not encoding or confidence < 0.8:
            raise utf8_error

        normalized_encoding = encoding.lower().replace("-", "_")
        if normalized_encoding in {"utf_16", "utf_16_le", "utf_16_be", "utf_32"}:
            raise utf8_error

        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            raise utf8_error


class TextLoaderParser(BaseDocumentParser):

    @classmethod
    def parser_id(cls) -> str:
        return "text_loader"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        text = _read_text(source)

        result: list[ParsedDocumentFragment] = []

        metadata = build_loader_metadata(
            source=source,
            parser=self.parser_id(),
            position=0,
        )

        result.append(
            ParsedDocumentFragment(
                text=text,
                metadata=metadata,
            )
        )

        return result
