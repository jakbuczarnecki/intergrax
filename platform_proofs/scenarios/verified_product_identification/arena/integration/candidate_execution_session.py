"""Reusable candidate execution session — one provider instance per stage evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidates import (
    EmbeddingArenaCandidate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    EmbeddingArenaCandidateSessionError,
    EmbeddingArenaEvaluationScopeError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputRole,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.embedding_port import (
    build_candidate_embedding_port,
    transform_texts_for_role,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    WarmupTimingSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.microbenchmark import (
    measure_warmup_timing,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    EmbeddingExecutionPort,
)

EmbeddingPortFactory = Callable[..., EmbeddingExecutionPort]


def _vectors_to_matrix(
    vectors: tuple[tuple[float, ...], ...],
    *,
    expected_dimension: int,
    label: str,
) -> NDArray[np.float64]:
    if not vectors:
        msg = f"{label} must not be empty"
        raise EmbeddingArenaEvaluationScopeError(msg)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        msg = f"{label} must be 2D, got ndim={matrix.ndim}"
        raise EmbeddingArenaEvaluationScopeError(msg)
    if matrix.shape[1] != expected_dimension:
        msg = (
            f"{label} dimension {matrix.shape[1]} != expected {expected_dimension}"
        )
        raise EmbeddingArenaEvaluationScopeError(msg)
    if not np.isfinite(matrix).all():
        msg = f"{label} contains non-finite values"
        raise EmbeddingArenaEvaluationScopeError(msg)
    return matrix


class EmbeddingArenaCandidateExecutionSession:
    """Owns exactly one embedding port for warmup, corpus, and query evaluation."""

    def __init__(
        self,
        candidate: EmbeddingArenaCandidate,
        *,
        provider_batch_size: int,
        device: str | None,
        port_factory: EmbeddingPortFactory = build_candidate_embedding_port,
    ) -> None:
        self._candidate = candidate
        self._provider_batch_size = provider_batch_size
        self._device = device
        self._port_factory = port_factory
        self._embedding: EmbeddingExecutionPort | None = None
        self._closed = False
        self._warmed = False

    def _ensure_open(self) -> EmbeddingExecutionPort:
        if self._closed:
            msg = "candidate execution session is closed"
            raise EmbeddingArenaCandidateSessionError(msg)
        if self._embedding is None:
            self._embedding = self._port_factory(
                self._candidate,
                provider_batch_size=self._provider_batch_size,
                device=self._device,
            )
        return self._embedding

    def warmup(self, canonical_texts: Sequence[str]) -> WarmupTimingSnapshot:
        embedding = self._ensure_open()
        transformed = transform_texts_for_role(
            self._candidate,
            role=EmbeddingInputRole.DOCUMENT,
            canonical_texts=canonical_texts,
        )
        timing = measure_warmup_timing(embedding, transformed)
        self._warmed = True
        return timing

    def embed_documents(
        self,
        canonical_texts: Sequence[str],
        *,
        expected_dimension: int,
    ) -> NDArray[np.float64]:
        embedding = self._ensure_open()
        transformed = transform_texts_for_role(
            self._candidate,
            role=EmbeddingInputRole.DOCUMENT,
            canonical_texts=canonical_texts,
        )
        vectors = embedding.embed_batch(transformed)
        return _vectors_to_matrix(
            vectors,
            expected_dimension=expected_dimension,
            label="document embeddings",
        )

    def embed_queries(
        self,
        query_texts: Sequence[str],
        *,
        expected_dimension: int,
    ) -> NDArray[np.float64]:
        embedding = self._ensure_open()
        transformed = transform_texts_for_role(
            self._candidate,
            role=EmbeddingInputRole.QUERY,
            canonical_texts=query_texts,
        )
        vectors = embedding.embed_batch(transformed)
        return _vectors_to_matrix(
            vectors,
            expected_dimension=expected_dimension,
            label="query embeddings",
        )

    def embed_query(
        self,
        query_text: str,
        *,
        expected_dimension: int,
    ) -> NDArray[np.float64]:
        matrix = self.embed_queries((query_text,), expected_dimension=expected_dimension)
        return matrix[0]

    def measure_query_latency(
        self,
        query_texts: Sequence[str],
        *,
        expected_dimension: int,
        repetitions: int = 5,
    ) -> tuple[float, float]:
        if not query_texts:
            msg = "query_texts must not be empty"
            raise ValueError(msg)
        if not self._warmed:
            msg = "session must be warmed before measuring query latency"
            raise EmbeddingArenaCandidateSessionError(msg)

        durations: list[float] = []
        for _ in range(repetitions):
            for query_text in query_texts[:3]:
                started = time.perf_counter()
                self.embed_query(query_text, expected_dimension=expected_dimension)
                durations.append(time.perf_counter() - started)

        ordered = sorted(durations)
        p50 = ordered[len(ordered) // 2]
        p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return p50, ordered[p95_index]

    def close(self) -> None:
        if self._closed:
            return
        if self._embedding is not None:
            self._embedding.close()
        self._embedding = None
        self._closed = True

    def __enter__(self) -> EmbeddingArenaCandidateExecutionSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_candidate_execution_session(
    candidate: EmbeddingArenaCandidate,
    *,
    provider_batch_size: int,
    device: str | None,
    port_factory: EmbeddingPortFactory = build_candidate_embedding_port,
) -> EmbeddingArenaCandidateExecutionSession:
    return EmbeddingArenaCandidateExecutionSession(
        candidate,
        provider_batch_size=provider_batch_size,
        device=device,
        port_factory=port_factory,
    )
