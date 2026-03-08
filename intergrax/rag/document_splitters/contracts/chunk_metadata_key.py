# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from enum import Enum


class ChunkMetadataKey(str, Enum):

    CHUNK_ID = "chunk_id"

    CHUNK_INDEX = "chunk_index"

    CHUNK_SIZE = "chunk_size"

    CHUNK_STRATEGY = "chunk_strategy"

    PARENT_CHUNK_ID = "parent_chunk_id"

    PARENT_CHUNK_INDEX = "parent_chunk_index"

    SECTION = "section"