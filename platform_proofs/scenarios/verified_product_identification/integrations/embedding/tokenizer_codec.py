"""Typed tokenizer codec resolution for canonical VPI embedding integration."""

from __future__ import annotations

from intergrax.rag.embedding.runtime.tokenizer_codec import resolve_hf_tokenizer_codec

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    EmbeddingTokenizerEncodeDecodePort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.ports import (
    DataPackEmbeddingPort,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)


def resolve_embedding_tokenizer_codec(
    embedding_port: DataPackEmbeddingPort,
) -> EmbeddingTokenizerEncodeDecodePort:
    if not isinstance(embedding_port, IntergraxEmbeddingBootstrapAdapter):
        raise VpiDataPackBuildError(
            "document embedding input policy requires IntergraxEmbeddingBootstrapAdapter"
        )
    try:
        return resolve_hf_tokenizer_codec(embedding_port.embedding_provider())
    except TypeError as exc:
        raise VpiDataPackBuildError(
            "canonical document embedding input policy requires HF tokenizer codec"
        ) from exc
