"""Gate 0 — embedding provider live compatibility before dense index bootstrap."""

from __future__ import annotations

import math

import numpy as np
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_registry
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderDependencyError,
    EmbeddingProviderRegistry,
)

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
    validate_resolved_provider_dimension,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)

_GATE0_PROBE_TEXTS: tuple[str, ...] = (
    "industrial relay module 24V DC",
    "stainless steel bolt M8 x 40mm",
    "wireless keyboard compact layout",
)


class RegistryEmbeddingReadinessProbe:
    """Resolve embedding provider through Intergrax registry and probe bounded texts."""

    def __init__(
        self,
        configuration: VpiEmbeddingConfiguration,
        *,
        registry: EmbeddingProviderRegistry | None = None,
        probe_texts: tuple[str, ...] = _GATE0_PROBE_TEXTS,
    ) -> None:
        self._configuration = configuration
        self._registry = registry
        self._probe_texts = probe_texts

    def probe(self) -> ValidationReport:
        model = self._configuration.model
        if model is None:
            raise VpiBootstrapProviderError("embedding model is required for Gate 0 probe")

        registry = self._registry or create_default_registry(embedding_model=model)
        try:
            provider = registry.get(self._configuration.provider)
        except (RuntimeError, EmbeddingProviderDependencyError) as exc:
            raise VpiBootstrapProviderError(
                f"embedding provider {self._configuration.provider!r} is unavailable: {exc}"
            ) from exc

        try:
            resolved_dimension = provider.dimension()
            validate_resolved_provider_dimension(
                configuration=self._configuration,
                resolved_dimension=resolved_dimension,
            )
            vectors = provider.embed(self._probe_texts)
        except Exception as exc:
            raise VpiBootstrapProviderError(
                f"embedding Gate 0 probe failed for provider={self._configuration.provider} "
                f"model={model}"
            ) from exc

        probe_result = _validate_probe_vectors(
            vectors=vectors,
            expected_dimension=self._configuration.expected_dimension,
            probe_count=len(self._probe_texts),
            provider=self._configuration.provider,
            model=model,
            resolved_dimension=resolved_dimension,
        )
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    name="embedding_gate0",
                    status=probe_result.status,
                    detail=probe_result.detail,
                ),
            )
        )

    def probe_detail(self) -> EmbeddingProbeResult:
        report = self.probe()
        check = report.checks[0]
        model = self._configuration.model or ""
        return EmbeddingProbeResult(
            status=check.status,
            provider=self._configuration.provider,
            model=model,
            resolved_dimension=self._configuration.expected_dimension,
            probe_vector_count=len(self._probe_texts),
            detail=check.detail,
        )


def _validate_probe_vectors(
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
