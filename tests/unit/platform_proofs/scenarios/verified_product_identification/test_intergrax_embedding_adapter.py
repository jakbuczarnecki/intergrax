"""Unit tests for VPI Intergrax embedding bootstrap adapter diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderRegistry,
)
from intergrax.rag.embedding.registry.execution_diagnostics import (
    EmbeddingProviderExecutionSnapshot,
    EmbeddingProviderExecutionSnapshotStatus,
)
from intergrax.rag.embedding.registry.profile import EmbeddingProfile
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)

pytestmark = pytest.mark.unit


class _DiagnosticsProvider(EmbeddingProvider):
    def provider_name(self) -> str:
        return "hf"

    def dimension(self) -> int:
        return 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=np.float32)

    def execution_snapshot(self) -> EmbeddingProviderExecutionSnapshot:
        return EmbeddingProviderExecutionSnapshot(
            configured_device="cuda",
            resolved_device="cuda:0",
            provider_batch_size=32,
            max_length=None,
            evidence_source="FakeProvider.execution_snapshot",
        )


class _PlainProvider(EmbeddingProvider):
    def provider_name(self) -> str:
        return "hf"

    def dimension(self) -> int:
        return 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=np.float32)


def _adapter_with_provider(provider: EmbeddingProvider) -> IntergraxEmbeddingBootstrapAdapter:
    configuration = VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model="test-model"),
        expected_dimension=2,
    )
    registry = EmbeddingProviderRegistry()
    registry.register(provider)
    return IntergraxEmbeddingBootstrapAdapter(configuration, registry=registry)


def test_execution_snapshot_available_from_diagnostics_capability() -> None:
    adapter = _adapter_with_provider(_DiagnosticsProvider())

    result = adapter.execution_snapshot()

    assert result.status is EmbeddingProviderExecutionSnapshotStatus.AVAILABLE
    assert result.snapshot is not None
    assert result.snapshot.resolved_device == "cuda:0"
    assert result.reason is None


def test_execution_snapshot_unavailable_without_diagnostics_capability() -> None:
    adapter = _adapter_with_provider(_PlainProvider())

    result = adapter.execution_snapshot()

    assert result.status is EmbeddingProviderExecutionSnapshotStatus.UNAVAILABLE
    assert result.snapshot is None
    assert result.reason == "provider_does_not_expose_execution_diagnostics"
