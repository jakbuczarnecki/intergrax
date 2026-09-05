"""Candidate embedding execution with role-specific input transformations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidates import (
    EmbeddingArenaCandidate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputRole,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.candidate_execution_session import (
    EmbeddingArenaCandidateExecutionSession,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.embedding_port import (
    transform_texts_for_role,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    build_candidate_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    WarmupTimingSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.microbenchmark import (
    run_provider_batch_candidate,
)


def embed_documents(
    candidate: EmbeddingArenaCandidate,
    *,
    canonical_texts: Sequence[str],
    provider_batch_size: int,
    device: str | None,
) -> NDArray[np.float64]:
    with EmbeddingArenaCandidateExecutionSession(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
    ) as session:
        return session.embed_documents(
            canonical_texts,
            expected_dimension=candidate.expected_dimension,
        )


def embed_query_vector(
    candidate: EmbeddingArenaCandidate,
    *,
    query_text: str,
    provider_batch_size: int,
    device: str | None,
) -> NDArray[np.float64]:
    with EmbeddingArenaCandidateExecutionSession(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
    ) as session:
        return session.embed_query(
            query_text,
            expected_dimension=candidate.expected_dimension,
        )


def measure_query_latency(
    candidate: EmbeddingArenaCandidate,
    *,
    query_texts: Sequence[str],
    provider_batch_size: int,
    device: str | None,
    repetitions: int = 5,
) -> tuple[float, float]:
    with EmbeddingArenaCandidateExecutionSession(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
    ) as session:
        session.warmup(query_texts[:1])
        return session.measure_query_latency(
            query_texts,
            expected_dimension=candidate.expected_dimension,
            repetitions=repetitions,
        )


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
    with EmbeddingArenaCandidateExecutionSession(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
    ) as session:
        return session.warmup(canonical_texts)
