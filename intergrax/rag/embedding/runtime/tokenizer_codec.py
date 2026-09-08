# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider


class EmbeddingTokenizerCodecPort(Protocol):
    def encode_text_tokens(self, text: str) -> tuple[int, ...]: ...

    def decode_text_tokens(self, token_ids: Sequence[int]) -> str: ...


def resolve_hf_tokenizer_codec(provider: EmbeddingProvider) -> EmbeddingTokenizerCodecPort:
    if not isinstance(provider, HFEmbeddingProvider):
        msg = "tokenizer codec resolution requires HFEmbeddingProvider"
        raise TypeError(msg)
    return provider
