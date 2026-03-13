# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter


@dataclass(slots=True)
class AnswerRequest:
    """
    Request object used by AnswerEngine.
    """

    query: str

    top_k: int = 5

    metadata_filter: MetadataFilter | None = None