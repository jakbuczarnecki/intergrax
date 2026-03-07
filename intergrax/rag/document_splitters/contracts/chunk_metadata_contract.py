# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from enum import Enum


class ChunkMetadataKey(str, Enum):

    # Unique identifier of the chunk
    CHUNK_ID = "chunk_id"

    # Sequential index of chunk within a source document
    CHUNK_INDEX = "chunk_index"

    # Name of the chunking strategy used
    CHUNK_STRATEGY = "chunk_strategy"

    # Original source document identifier
    SOURCE_DOCUMENT_ID = "source_document_id"

    # Character offset of chunk start within the source document
    START_OFFSET = "start_offset"

    # Character offset of chunk end within the source document
    END_OFFSET = "end_offset"

    # Parent chunk identifier (used for hierarchical chunking)
    PARENT_CHUNK_ID = "parent_chunk_id"

    PARENT_CHUNK_INDEX = "parent_chunk_index"

    # Section identifier extracted from document structure
    SECTION = "section"

    # Page number (if available)
    PAGE_NUMBER = "page"

    # Total number of chunks produced from the source document
    TOTAL_CHUNKS = "total_chunks"