# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, List, Optional, cast

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import (
    BaseChunkingStrategy,
)
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey


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
        from docling_core.types.doc import (
            CodeItem,
            FormulaItem,
            ListItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
        )

        chunks: List[Document] = []

        for document in documents:

            metadata = dict(document.metadata)

            docling_doc: Optional[DoclingDocument] = cast(
                Optional[DoclingDocument],
                metadata.get(DocumentMetadataKey.DOCLING_DOCUMENT_META),
            )

            if docling_doc is None:
                continue

            buffer: List[str] = []
            buffer_length: int = 0
            chunk_index = 0
            current_section: Optional[str] = None

            for item, _level in docling_doc.iterate_items():

                if isinstance(item, SectionHeaderItem):

                    current_section = item.text

                    if buffer:
                        chunks.append(
                            self._create_chunk(
                                "\n".join(buffer),
                                metadata,
                                chunk_index,
                                current_section,
                            )
                        )
                        chunk_index = len(chunks)
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
                                "\n".join(buffer),
                                metadata,
                                chunk_index,
                                current_section,
                            )
                        )
                        chunk_index = len(chunks)
                        buffer = []
                        buffer_length = 0

                    table_text = item.export_to_markdown()

                    chunks.append(
                        self._create_single_chunk(
                            table_text,
                            metadata,
                            chunk_index,
                            current_section,
                        )
                    )
                    chunk_index = len(chunks)
                    continue

                elif isinstance(item, PictureItem):

                    caption = item.label or ""

                    chunks.append(
                        self._create_single_chunk(
                            caption,
                            metadata,
                            chunk_index,
                            current_section,
                        )
                    )
                    chunk_index = len(chunks)
                    continue

                elif isinstance(item, CodeItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                elif isinstance(item, FormulaItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                if buffer_length >= self.MAX_CHUNK_SIZE:

                    chunks.append(
                        self._create_chunk(
                            "\n".join(buffer),
                            metadata,
                            chunk_index,
                            current_section,
                        )
                    )

                    chunk_index = len(chunks)
                    buffer = []
                    buffer_length = 0

            if buffer:

                chunks.append(
                    self._create_chunk(
                        "\n".join(buffer),
                        metadata,
                        chunk_index,
                        current_section,
                    )
                )

        return chunks

    def _create_chunk(
        self,
        chunk_text: str,
        metadata: dict,
        chunk_index: int,
        section: Optional[str],
    ) -> Document:

        chunk_metadata = dict(metadata)

        chunk_metadata[ChunkMetadataKey.CHUNK_INDEX] = chunk_index
        chunk_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()
        chunk_metadata[ChunkMetadataKey.CHUNK_SIZE] = len(chunk_text)

        if section:
            chunk_metadata[ChunkMetadataKey.SECTION] = section

        document_id = chunk_metadata.get(DocumentMetadataKey.DOCUMENT_ID)

        if document_id:
            chunk_metadata[ChunkMetadataKey.CHUNK_ID] = (
                f"{document_id}:{self.strategy_id()}:{chunk_index}"
            )

        chunk_metadata.pop(DocumentMetadataKey.DOCLING_DOCUMENT_META, None)

        return Document(
            page_content=chunk_text,
            metadata=chunk_metadata,
        )

    def _create_single_chunk(
        self,
        text: str,
        metadata: dict,
        chunk_index: int,
        section: Optional[str],
    ) -> Document:

        chunk_metadata = dict(metadata)

        chunk_metadata[ChunkMetadataKey.CHUNK_INDEX] = chunk_index
        chunk_metadata[ChunkMetadataKey.CHUNK_STRATEGY] = self.strategy_id()
        chunk_metadata[ChunkMetadataKey.CHUNK_SIZE] = len(text)

        if section:
            chunk_metadata[ChunkMetadataKey.SECTION] = section

        document_id = chunk_metadata.get(DocumentMetadataKey.DOCUMENT_ID)

        if document_id:
            chunk_metadata[ChunkMetadataKey.CHUNK_ID] = (
                f"{document_id}:{self.strategy_id()}:{chunk_index}"
            )

        chunk_metadata.pop(DocumentMetadataKey.DOCLING_DOCUMENT_META, None)

        return Document(
            page_content=text,
            metadata=chunk_metadata,
        )