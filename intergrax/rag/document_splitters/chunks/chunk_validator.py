# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Any

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import (
    ChunkMetadataKey,
)


class ChunkValidator:
    """
    Validates structural invariants of generated chunks.
    """

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> None:
        """
        Validate chunk metadata invariants.

        Raises
        ------
        ValueError
            If required metadata fields are missing.
        """

        required_keys = (
            ChunkMetadataKey.CHUNK_INDEX.value,
            ChunkMetadataKey.CHUNK_STRATEGY.value,
            ChunkMetadataKey.CHUNK_ID.value,
        )

        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"Chunk metadata missing required key: {key}")