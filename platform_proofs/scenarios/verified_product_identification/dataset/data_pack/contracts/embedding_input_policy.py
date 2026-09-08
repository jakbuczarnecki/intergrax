"""Document embedding input policy contracts for VPI data packs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET = 768
VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION = (
    "vpi-bge-m3-document-token-budget-768-v1"
)


class EmbeddingTokenizerEncodeDecodePort(Protocol):
    """Provider-neutral tokenizer encode/decode seam for document input policies."""

    def encode_text_tokens(self, text: str) -> tuple[int, ...]: ...

    def decode_text_tokens(self, token_ids: Sequence[int]) -> str: ...


class DataPackDocumentEmbeddingInputPolicyPort(Protocol):
    """Deterministic transformation from full canonical semantic_text to embedding input."""

    @property
    def policy_version(self) -> str: ...

    def apply_document(self, text: str) -> str: ...
