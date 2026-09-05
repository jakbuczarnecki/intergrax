"""Candidate embedding execution with role-specific input transformations."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

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
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    WarmupTimingSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.microbenchmark import (
    build_embedding_execution_port,
    measure_warmup_timing,
    run_provider_batch_candidate,
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
    transformation = resolve_input_transformation(candidate.query_instruction_policy.policy_id)
    return tuple(transformation.transform(role, text) for text in canonical_texts)


def embed_documents(
    candidate: EmbeddingArenaCandidate,
    *,
    canonical_texts: Sequence[str],
    provider_batch_size: int,
    device: str | None,
) -> NDArray[np.float64]:
    transformed = transform_texts_for_role(
        candidate,
        role=EmbeddingInputRole.DOCUMENT,
        canonical_texts=canonical_texts,
    )
    embedding = build_candidate_embedding_port(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
    )
    try:
        embedding.probe()
        vectors = embedding.embed_batch(transformed)
    finally:
        embedding.close()
    matrix = np.asarray(vectors, dtype=np.float64)
    return matrix


def embed_query_vector(
    candidate: EmbeddingArenaCandidate,
    *,
    query_text: str,
    provider_batch_size: int,
    device: str | None,
) -> NDArray[np.float64]:
    transformed = transform_texts_for_role(
        candidate,
        role=EmbeddingInputRole.QUERY,
        canonical_texts=(query_text,),
    )[0]
    embedding = build_candidate_embedding_port(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
    )
    try:
        vectors = embedding.embed_batch((transformed,))
    finally:
        embedding.close()
    return np.asarray(vectors[0], dtype=np.float64)


def measure_query_latency(
    candidate: EmbeddingArenaCandidate,
    *,
    query_texts: Sequence[str],
    provider_batch_size: int,
    device: str | None,
    repetitions: int = 5,
) -> tuple[float, float]:
    if not query_texts:
        msg = "query_texts must not be empty"
        raise ValueError(msg)
    durations: list[float] = []
    for _ in range(repetitions):
        for query_text in query_texts[:3]:
            started = time.perf_counter()
            embed_query_vector(
                candidate,
                query_text=query_text,
                provider_batch_size=provider_batch_size,
                device=device,
            )
            durations.append(time.perf_counter() - started)
    ordered = sorted(durations)
    p50 = ordered[len(ordered) // 2]
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return p50, ordered[p95_index]


def run_candidate_microbenchmark(
    candidate: EmbeddingArenaCandidate,
    canonical_texts: Sequence[str],
    *,
    provider_batch_size: int,
    device: str | None,
):
    configuration = build_candidate_embedding_configuration(candidate)
    transformed = transform_texts_for_role(
        candidate,
        role=EmbeddingInputRole.DOCUMENT,
        canonical_texts=canonical_texts,
    )
    return run_provider_batch_candidate(
        configuration,
        transformed,
        provider_batch_size=provider_batch_size,
        device=device,
        expected_dimension=candidate.expected_dimension,
    )


def measure_candidate_warmup(
    candidate: EmbeddingArenaCandidate,
    canonical_texts: Sequence[str],
    *,
    provider_batch_size: int,
    device: str | None,
) -> WarmupTimingSnapshot:
    embedding = build_candidate_embedding_port(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
    )
    transformed = transform_texts_for_role(
        candidate,
        role=EmbeddingInputRole.DOCUMENT,
        canonical_texts=canonical_texts,
    )
    try:
        return measure_warmup_timing(embedding, transformed)
    finally:
        embedding.close()
