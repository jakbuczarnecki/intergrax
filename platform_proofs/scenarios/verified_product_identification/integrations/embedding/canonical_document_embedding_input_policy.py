"""Composition for canonical BGE-M3 768-token document embedding input policy."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.token_budget_document_embedding_input_policy import (
    TokenBudgetDocumentEmbeddingInputPolicy,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
    VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
    DataPackDocumentEmbeddingInputPolicyPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.ports import (
    DataPackEmbeddingPort,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.tokenizer_codec import (
    resolve_embedding_tokenizer_codec,
)


def resolve_canonical_document_embedding_input_policy(
    embedding_port: DataPackEmbeddingPort,
) -> DataPackDocumentEmbeddingInputPolicyPort:
    tokenizer_codec = resolve_embedding_tokenizer_codec(embedding_port)
    return TokenBudgetDocumentEmbeddingInputPolicy(
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
        tokenizer_codec=tokenizer_codec,
    )
