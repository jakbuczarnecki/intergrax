"""Intergrax embedding bootstrap adapter — single provider instance per run."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_registry
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderDependencyError,
    EmbeddingProviderRegistry,
)
from intergrax.rag.embedding.registry.execution_diagnostics import (
    EmbeddingProviderExecutionDiagnostics,
    EmbeddingProviderExecutionSnapshotResult,
    EmbeddingProviderExecutionSnapshotStatus,
)

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
    validate_resolved_provider_dimension,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.validation.embedding_probe_validation import (
    GATE0_PROBE_TEXTS,
    validate_probe_vectors,
)


class IntergraxEmbeddingBootstrapAdapter:
    """Reference ``EmbeddingExecutionPort`` — owns one resolved provider for probe + ingest."""

    def __init__(
        self,
        configuration: VpiEmbeddingConfiguration,
        *,
        registry: EmbeddingProviderRegistry | None = None,
        probe_texts: tuple[str, ...] = GATE0_PROBE_TEXTS,
        execution_configuration: VpiEmbeddingProviderExecutionConfiguration | None = None,
    ) -> None:
        self._configuration = configuration
        self._probe_texts = probe_texts
        model = configuration.model
        if model is None:
            raise VpiBootstrapProviderError("embedding model is required")
        execution_config = (
            execution_configuration.execution
            if execution_configuration is not None
            else None
        )
        resolved_registry = registry or create_default_registry(
            embedding_model=model,
            execution_config=execution_config,
        )
        try:
            self._provider: EmbeddingProvider = resolved_registry.get(configuration.provider)
        except (RuntimeError, EmbeddingProviderDependencyError) as exc:
            raise VpiBootstrapProviderError(
                f"embedding provider {configuration.provider!r} is unavailable: {exc}"
            ) from exc

    def probe(self) -> EmbeddingProbeResult:
        model = self._configuration.model
        if model is None:
            raise VpiBootstrapProviderError("embedding model is required for Gate 0 probe")
        try:
            resolved_dimension = self._provider.dimension()
            validate_resolved_provider_dimension(
                configuration=self._configuration,
                resolved_dimension=resolved_dimension,
            )
            vectors = self._provider.embed(list(self._probe_texts))
        except Exception as exc:
            raise VpiBootstrapProviderError(
                f"embedding Gate 0 probe failed for provider={self._configuration.provider} "
                f"model={model}"
            ) from exc

        return validate_probe_vectors(
            vectors=vectors,
            expected_dimension=self._configuration.expected_dimension,
            probe_count=len(self._probe_texts),
            provider=self._configuration.provider,
            model=model,
            resolved_dimension=resolved_dimension,
        )

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        try:
            vectors = self._provider.embed(list(texts))
        except Exception as exc:
            raise VpiBootstrapProviderError("embedding batch failed") from exc
        if vectors.ndim != 2:
            raise VpiBootstrapProviderError(
                f"expected 2D embedding array, got ndim={vectors.ndim}"
            )
        if vectors.shape[0] != len(texts):
            raise VpiBootstrapProviderError(
                f"expected {len(texts)} vectors, got {vectors.shape[0]}"
            )
        if not np.isfinite(vectors).all():
            raise VpiBootstrapProviderError("embedding batch contains non-finite values")
        rows: list[tuple[float, ...]] = []
        for row_index in range(vectors.shape[0]):
            row = vectors[row_index]
            if row.size == 0 or math.isclose(float(np.linalg.norm(row)), 0.0):
                raise VpiBootstrapProviderError(
                    f"embedding vector at index {row_index} is empty or zero-norm"
                )
            rows.append(tuple(float(value) for value in row))
        return tuple(rows)

    def close(self) -> None:
        return None

    def execution_snapshot(self) -> EmbeddingProviderExecutionSnapshotResult:
        if isinstance(self._provider, EmbeddingProviderExecutionDiagnostics):
            self._provider.dimension()
            return EmbeddingProviderExecutionSnapshotResult(
                status=EmbeddingProviderExecutionSnapshotStatus.AVAILABLE,
                snapshot=self._provider.execution_snapshot(),
                reason=None,
            )
        return EmbeddingProviderExecutionSnapshotResult(
            status=EmbeddingProviderExecutionSnapshotStatus.UNAVAILABLE,
            snapshot=None,
            reason="provider_does_not_expose_execution_diagnostics",
        )
