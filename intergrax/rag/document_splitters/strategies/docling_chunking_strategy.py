# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence, List, Optional, cast

from langchain_core.documents import Document

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    SectionHeaderItem,
    TextItem,
    ListItem,
    TableItem,
    PictureItem,
    CodeItem,
    FormulaItem,
)

from intergrax.rag.document_loaders.contracts.metadata_constants import DOCLING_DOCUMENT_META_KEY
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import (
    BaseChunkingStrategy,
)
from intergrax.rag.document_splitters.contracts.chunk_metadata_contract import ChunkMetadataKey


class DoclingChunkingStrategy(BaseChunkingStrategy):
    """
    Production-grade Docling AST chunking strategy.

    The strategy consumes DoclingDocument stored in metadata and
    creates chunks aligned with structural elements of the document.
    """

    MAX_CHUNK_SIZE = 1500

    @classmethod
    def strategy_id(cls) -> str:
        return "docling"

    def chunk(
        self,
        documents: Sequence[Document],
    ) -> Sequence[Document]:

        chunks: List[Document] = []

        for document in documents:

            metadata = dict(document.metadata)

            docling_doc: Optional[DoclingDocument] = cast(
                Optional[DoclingDocument],
                metadata.get(DOCLING_DOCUMENT_META_KEY),
            )

            if docling_doc is None:
                continue

            buffer: List[str] = []
            buffer_length: int = 0
            chunk_index = 0

            for item, _level in docling_doc.iterate_items():

                if isinstance(item, SectionHeaderItem):

                    if buffer:
                        chunks.append(
                            self._create_chunk(
                                buffer,
                                metadata,
                                chunk_index,
                            )
                        )
                        chunk_index += 1
                        buffer = []
                        buffer_length = 0

                    buffer.append(item.text)
                    buffer_length += len(item.text)

                elif isinstance(item, TextItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                elif isinstance(item, ListItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                elif isinstance(item, TableItem):

                    if buffer:
                        chunks.append(
                            self._create_chunk(
                                buffer,
                                metadata,
                                chunk_index,
                            )
                        )
                        chunk_index += 1
                        buffer = []
                        buffer_length = 0

                    table_text = item.export_to_markdown()

                    chunks.append(
                        self._create_single_chunk(
                            table_text,
                            metadata,
                            chunk_index,
                        )
                    )
                    chunk_index += 1

                elif isinstance(item, PictureItem):

                    caption = item.label or ""

                    chunks.append(
                        self._create_single_chunk(
                            caption,
                            metadata,
                            chunk_index,
                        )
                    )
                    chunk_index += 1

                elif isinstance(item, CodeItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                elif isinstance(item, FormulaItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                if sum(len(x) for x in buffer) >= self.MAX_CHUNK_SIZE:

                    chunks.append(
                        self._create_chunk(
                            buffer,
                            metadata,
                            chunk_index,
                        )
                    )

                    chunk_index += 1
                    buffer = []
                    buffer_length = 0

            if buffer:

                chunks.append(
                    self._create_chunk(
                        buffer,
                        metadata,
                        chunk_index,
                    )
                )

        return chunks

    def _create_chunk(
        self,
        buffer: List[str],
        metadata: dict,
        chunk_index: int,
    ) -> Document:

        chunk_text = "\n".join(buffer)

        chunk_metadata = dict(metadata)

        chunk_metadata[ChunkMetadataKey.CHUNK_INDEX] = chunk_index
        chunk_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()

        chunk_metadata.pop(DOCLING_DOCUMENT_META_KEY, None)

        return Document(
            page_content=chunk_text,
            metadata=chunk_metadata,
        )

    def _create_single_chunk(
        self,
        text: str,
        metadata: dict,
        chunk_index: int,
    ) -> Document:

        chunk_metadata = dict(metadata)
        chunk_metadata[ChunkMetadataKey.CHUNK_INDEX] = chunk_index
        chunk_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()

        chunk_metadata.pop(DOCLING_DOCUMENT_META_KEY, None)

        return Document(
            page_content=text,
            metadata=chunk_metadata,
        )