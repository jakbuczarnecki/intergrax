"""Embedding port construction and role-specific text transformation."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    build_candidate_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.input_policies import (
    resolve_input_transformation,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidates import (
    EmbeddingArenaCandidate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputRole,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)


def build_candidate_embedding_port(
    candidate: EmbeddingArenaCandidate,
    *,
    provider_batch_size: int,
    device: str | None,
) -> IntergraxEmbeddingBootstrapAdapter:
    configuration = build_candidate_embedding_configuration(candidate)
    execution_configuration = VpiEmbeddingProviderExecutionConfiguration(
        execution=EmbeddingProviderExecutionConfig(
            device=device,
            batch_size=provider_batch_size,
        )
    )
    return IntergraxEmbeddingBootstrapAdapter(
        configuration,
        execution_configuration=execution_configuration,
    )


def transform_texts_for_role(
    candidate: EmbeddingArenaCandidate,
    *,
    role: EmbeddingInputRole,
    canonical_texts: Sequence[str],
) -> tuple[str, ...]:
    if role is EmbeddingInputRole.QUERY:
        policy_id = candidate.query_instruction_policy.policy_id
    else:
        policy_id = candidate.document_instruction_policy.policy_id
    transformation = resolve_input_transformation(policy_id)
    return tuple(transformation.transform(role, text) for text in canonical_texts)
