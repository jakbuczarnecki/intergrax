# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.rerankers.providers._cross_encoder_base import (
    _CrossEncoderBaseReranker,
)


class CrossEncoderReranker(_CrossEncoderBaseReranker):

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def name(cls) -> str:
        return "cross_encoder"