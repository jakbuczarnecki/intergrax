"""Shared Gate 0 probe validation helpers."""

from __future__ import annotations

import math

import numpy as np

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationStatus,
)

GATE0_PROBE_TEXTS: tuple[str, ...] = (
    "industrial relay module 24V DC",
    "stainless steel bolt M8 x 40mm",
    "wireless keyboard compact layout",
)


def validate_probe_vectors(
    *,
    vectors: np.ndarray,
    expected_dimension: int,
    probe_count: int,
    provider: str,
    model: str,
    resolved_dimension: int,
) -> EmbeddingProbeResult:
    if vectors.ndim != 2:
        return EmbeddingProbeResult(
            status=ValidationStatus.FAIL,
            provider=provider,
            model=model,
            resolved_dimension=resolved_dimension,
            probe_vector_count=probe_count,
            detail=f"expected 2D embedding array, got ndim={vectors.ndim}",
        )
    if vectors.shape[0] != probe_count:
        return EmbeddingProbeResult(
            status=ValidationStatus.FAIL,
            provider=provider,
            model=model,
            resolved_dimension=resolved_dimension,
            probe_vector_count=probe_count,
            detail=f"expected {probe_count} vectors, got {vectors.shape[0]}",
        )
    if vectors.shape[1] != expected_dimension:
        return EmbeddingProbeResult(
            status=ValidationStatus.FAIL,
            provider=provider,
            model=model,
            resolved_dimension=resolved_dimension,
            probe_vector_count=probe_count,
            detail=(
                f"probe vector dimension {vectors.shape[1]} != expected {expected_dimension}"
            ),
        )
    if not np.isfinite(vectors).all():
        return EmbeddingProbeResult(
            status=ValidationStatus.FAIL,
            provider=provider,
            model=model,
            resolved_dimension=resolved_dimension,
            probe_vector_count=probe_count,
            detail="probe vectors contain non-finite values",
        )
    for row_index in range(vectors.shape[0]):
        row = vectors[row_index]
        if row.size == 0 or math.isclose(float(np.linalg.norm(row)), 0.0):
            return EmbeddingProbeResult(
                status=ValidationStatus.FAIL,
                provider=provider,
                model=model,
                resolved_dimension=resolved_dimension,
                probe_vector_count=probe_count,
                detail=f"probe vector at index {row_index} is empty or zero-norm",
            )

    return EmbeddingProbeResult(
        status=ValidationStatus.PASS,
        provider=provider,
        model=model,
        resolved_dimension=resolved_dimension,
        probe_vector_count=probe_count,
        detail="embedding Gate 0 probe passed",
    )
