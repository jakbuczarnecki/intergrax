"""Unit tests for VPI Intergrax embedding bootstrap adapter diagnostics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
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
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapProviderError,
)

pytestmark = pytest.mark.unit

ADAPTER_PATH = Path(
    "platform_proofs/scenarios/verified_product_identification/integrations/embedding/intergrax_adapter.py"
)


@pytest.fixture(autouse=True)
def _bootstrap_embedding_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


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


class _MismatchedProvider(EmbeddingProvider):
    def provider_name(self) -> str:
        return "openai"

    def dimension(self) -> int:
        return 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=np.float32)


def _adapter_with_provider(provider: EmbeddingProvider) -> IntergraxEmbeddingBootstrapAdapter:
    configuration = VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model="test-model"),
        expected_dimension=2,
    )
    return IntergraxEmbeddingBootstrapAdapter(configuration, provider=provider)


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


def test_injected_provider_matching_identity_is_accepted() -> None:
    adapter = _adapter_with_provider(_PlainProvider())

    assert adapter.execution_snapshot().status is EmbeddingProviderExecutionSnapshotStatus.UNAVAILABLE


def test_injected_provider_mismatch_raises_vpi_bootstrap_provider_error() -> None:
    configuration = VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model="test-model"),
        expected_dimension=2,
    )

    with pytest.raises(VpiBootstrapProviderError, match="configured provider='hf'"):
        IntergraxEmbeddingBootstrapAdapter(configuration, provider=_MismatchedProvider())


def test_canonical_binder_path_delegates_to_bind_embedding_provider() -> None:
    configuration = VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model="test-model"),
        expected_dimension=2,
    )
    sentinel = _PlainProvider()

    with patch(
        "platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter.bind_embedding_provider",
        return_value=sentinel,
    ) as bind_mock:
        adapter = IntergraxEmbeddingBootstrapAdapter(configuration)

    bind_mock.assert_called_once()
    kwargs = bind_mock.call_args.kwargs
    assert kwargs["integration_profile"] == IntegrationProfile(embedding_provider="hf")
    assert kwargs["embedding_profile"] == EmbeddingProfile(provider="hf", model="test-model")
    assert adapter.execution_snapshot().status is EmbeddingProviderExecutionSnapshotStatus.UNAVAILABLE


def test_adapter_source_has_no_registry_api() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")

    assert "EmbeddingProviderRegistry" not in source
    assert "create_default_registry" not in source
    assert "provider_factory_registration" not in source
