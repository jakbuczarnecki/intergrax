# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib


class ChunkIDGenerator:
    """
    Deterministic chunk identifier generator.

    The identifier must remain stable across:
    - repeated ingestion
    - distributed pipelines
    - different workers
    """

    @staticmethod
    def generate(
        document_id: str,
        strategy_id: str,
        chunk_index: int,
        text: str,
    ) -> str:

        base = f"{document_id}|{strategy_id}|{chunk_index}|{text}"

        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()

        return digest[:16]