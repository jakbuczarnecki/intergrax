# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata



class TextLoaderParser(BaseDocumentParser):

    @classmethod
    def parser_id(cls) -> str:
        return "text_loader"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        loader = TextLoader(
            source,
            autodetect_encoding=True
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