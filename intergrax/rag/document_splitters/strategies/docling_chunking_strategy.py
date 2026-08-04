# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.compat.legacy_runtime_document import get_parser_native_handle
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import (
    BaseChunkingStrategy,
)
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


class DoclingChunkingStrategy(BaseChunkingStrategy):
    """
    Production-grade Docling AST chunking strategy.

    The strategy consumes DoclingDocument from private parser runtime state and
    creates chunks aligned with structural elements of the document.
    """

    MAX_CHUNK_SIZE = 1500

    @classmethod
    def strategy_id(cls) -> str:
        return "docling"

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[KnowledgeDocument]:
        from docling_core.types.doc import (
            CodeItem,
            FormulaItem,
            ListItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
        )

        chunks: list[KnowledgeDocument] = []

        for document in documents:
            docling_doc = get_parser_native_handle(document)
            if docling_doc is None:
                continue

            docling_document: DoclingDocument = docling_doc  # type: ignore[assignment]

            buffer: list[str] = []
            buffer_length: int = 0
            chunk_index = 0
            current_section: str | None = None

            for item, _level in docling_document.iterate_items():
                if isinstance(item, SectionHeaderItem):
                    if buffer:
                        chunk = self._try_create_chunk(
                            document,
                            "\n".join(buffer),
                            chunk_index,
                            current_section,
                        )
                        if chunk is not None:
                            chunks.append(chunk)
                            chunk_index += 1
                        buffer = []
                        buffer_length = 0

                    current_section = item.text
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
                        chunk = self._try_create_chunk(
                            document,
                            "\n".join(buffer),
                            chunk_index,
                            current_section,
                        )
                        if chunk is not None:
                            chunks.append(chunk)
                            chunk_index += 1
                        buffer = []
                        buffer_length = 0

                    table_text = item.export_to_markdown()
                    chunk = self._try_create_chunk(
                        document,
                        table_text,
                        chunk_index,
                        current_section,
                    )
                    if chunk is not None:
                        chunks.append(chunk)
                        chunk_index += 1
                    continue

                elif isinstance(item, PictureItem):
                    caption = item.label or ""
                    chunk = self._try_create_chunk(
                        document,
                        caption,
                        chunk_index,
                        current_section,
                    )
                    if chunk is not None:
                        chunks.append(chunk)
                        chunk_index += 1
                    continue

                elif isinstance(item, CodeItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                elif isinstance(item, FormulaItem):
                    buffer.append(item.text)
                    buffer_length += len(item.text)

                if buffer_length >= self.MAX_CHUNK_SIZE:
                    chunk = self._try_create_chunk(
                        document,
                        "\n".join(buffer),
                        chunk_index,
                        current_section,
                    )
                    if chunk is not None:
                        chunks.append(chunk)
                        chunk_index += 1
                    buffer = []
                    buffer_length = 0

            if buffer:
                chunk = self._try_create_chunk(
                    document,
                    "\n".join(buffer),
                    chunk_index,
                    current_section,
                )
                if chunk is not None:
                    chunks.append(chunk)

        return chunks

    def _try_create_chunk(
        self,
        source_document: KnowledgeDocument,
        chunk_text: str,
        chunk_index: int,
        section: str | None,
    ) -> KnowledgeDocument | None:
        if not chunk_text.strip():
            return None

        metadata_updates: dict[str, object] = {}
        if section:
            metadata_updates[ChunkMetadataKey.SECTION.value] = section

        return build_derived_chunk(
            source_document,
            content=chunk_text,
            strategy_id=self.strategy_id(),
            chunk_index=chunk_index,
            metadata_updates=metadata_updates or None,
        )
