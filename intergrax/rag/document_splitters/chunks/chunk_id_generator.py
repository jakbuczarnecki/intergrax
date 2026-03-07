# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib


class ChunkIDGenerator:
    """
    Deterministic chunk identifier generator.

    This generator produces a stable SHA256 identifier for each chunk.
    The identifier is based on the source identifier, chunk index,
    and normalized chunk text.

    The goal is to ensure stable chunk IDs across repeated ingestions
    of the same document.
    """

    @staticmethod
    def generate(
        source: str,
        chunk_index: int,
        text: str,
    ) -> str:
        """
        Generate a deterministic chunk identifier.

        Parameters
        ----------
        source:
            Unique identifier of the source document.
        chunk_index:
            Sequential index of the chunk within the document.
        text:
            Chunk text content.

        Returns
        -------
        str
            Stable SHA256 chunk identifier.
        """

        normalized_text = text.strip()

        payload = f"{source}|{chunk_index}|{normalized_text}"

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()