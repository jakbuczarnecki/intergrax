# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy


class RecursiveChunkingStrategy(BaseChunkingStrategy):
    """
    Deterministic recursive chunking strategy.

    Splits documents recursively at increasingly finer textual boundaries,
    then adds bounded character overlap between adjacent chunks.
    """

    _SEPARATORS: tuple[str, ...] = (
        "\r\n\r\n",
        "\n\n",
        "\r\n",
        "\n",
        ". ",
        "? ",
        "! ",
        " ",
    )

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if type(chunk_overlap) is not int or chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @classmethod
    def strategy_id(cls) -> str:
        return "recursive"

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[KnowledgeDocument]:

        chunks: list[KnowledgeDocument] = []

        for doc in documents:
            text = doc.content
            chunk_index = 0

            if not text.strip():
                continue

            if len(text) <= self._chunk_size:
                source_ranges = [(0, len(text))]
            else:
                payload_size = self._chunk_size - self._chunk_overlap
                source_ranges = self._recursive_ranges(
                    text,
                    start=0,
                    end=len(text),
                    limit=payload_size,
                    separators=self._SEPARATORS,
                )

            for source_start, source_end in source_ranges:
                if not text[source_start:source_end].strip():
                    continue

                chunk_start = max(0, source_start - self._chunk_overlap)
                chunk_text = text[chunk_start:source_end]
                if not chunk_text.strip():
                    continue

                chunks.append(
                    build_derived_chunk(
                        doc,
                        content=chunk_text,
                        strategy_id=self.strategy_id(),
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        return chunks

    @classmethod
    def _recursive_ranges(
        cls,
        text: str,
        *,
        start: int,
        end: int,
        limit: int,
        separators: Sequence[str],
    ) -> list[tuple[int, int]]:
        """Return contiguous, lossless ranges no longer than ``limit``."""
        if end - start <= limit:
            return [(start, end)]

        separator_index = next(
            (
                index
                for index, separator in enumerate(separators)
                if text.find(separator, start, end) != -1
            ),
            None,
        )
        if separator_index is None:
            return [
                (position, min(position + limit, end))
                for position in range(start, end, limit)
            ]

        separator = separators[separator_index]
        parts: list[tuple[int, int]] = []
        part_start = start
        search_start = start
        separator_length = len(separator)

        while search_start < end:
            separator_start = text.find(separator, search_start, end)
            if separator_start == -1:
                break

            part_end = separator_start + separator_length
            if part_end > part_start:
                parts.append((part_start, part_end))
                part_start = part_end
            search_start = part_end

        if part_start < end:
            parts.append((part_start, end))

        ranges: list[tuple[int, int]] = []
        finer_separators = separators[separator_index + 1 :]
        for part_start, part_end in parts:
            ranges.extend(
                cls._recursive_ranges(
                    text,
                    start=part_start,
                    end=part_end,
                    limit=limit,
                    separators=finer_separators,
                )
            )

        return cls._merge_ranges(ranges, limit)

    @staticmethod
    def _merge_ranges(
        ranges: Sequence[tuple[int, int]],
        limit: int,
    ) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []

        for start, end in ranges:
            if not merged:
                merged.append((start, end))
                continue

            previous_start, previous_end = merged[-1]
            if start == previous_end and end - previous_start <= limit:
                merged[-1] = (previous_start, end)
            else:
                merged.append((start, end))

        return merged
