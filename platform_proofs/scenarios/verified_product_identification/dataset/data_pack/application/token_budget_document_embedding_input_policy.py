"""Tokenizer token-budget document embedding input policy implementation."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.token_budget_truncation import (
    truncate_to_token_limit,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    EmbeddingTokenizerEncodeDecodePort,
)


@dataclass(frozen=True, slots=True)
class TokenBudgetDocumentEmbeddingInputPolicy:
    """Immutable tokenizer-bounded document embedding input policy."""

    policy_version: str
    token_budget: int
    tokenizer_codec: EmbeddingTokenizerEncodeDecodePort

    def apply_document(self, text: str) -> str:
        return truncate_to_token_limit(
            text,
            token_limit=self.token_budget,
            encode=self.tokenizer_codec.encode_text_tokens,
            decode=self.tokenizer_codec.decode_text_tokens,
        )
