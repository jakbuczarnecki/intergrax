# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_community.document_loaders import TextLoader

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata


class TextLoaderParser(BaseDocumentParser):

    @classmethod
    def parser_id(cls) -> str:
        return "text_loader"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:

        loader = TextLoader(
            source,
            autodetect_encoding=True
        )

        docs = loader.load()

        result: list[ParsedDocumentFragment] = []

        for i, d in enumerate(docs):

            metadata = build_loader_metadata(
                source=source,
                parser=self.parser_id(),
                position=i,
            )

            metadata.update(d.metadata or {})

            result.append(
                ParsedDocumentFragment(
                    text=d.page_content,
                    metadata=metadata,
                )
            )

        return result
