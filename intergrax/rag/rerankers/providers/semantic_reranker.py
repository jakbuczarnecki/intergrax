# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.rerankers.providers._cross_encoder_base import (
    _CrossEncoderBaseReranker,
)


class SemanticReranker(_CrossEncoderBaseReranker):

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    @classmethod
    def name(cls) -> str:
        return "semantic"