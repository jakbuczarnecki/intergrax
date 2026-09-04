"""Unit tests for VPI embedding provider execution configuration."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingDeviceUnavailableError,
    VpiEmbeddingProviderExecutionConfiguration,
    assert_execution_device_available,
    load_vpi_embedding_provider_execution_configuration,
)
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

pytestmark = pytest.mark.unit


def test_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPI_EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("VPI_EMBEDDING_PROVIDER_BATCH_SIZE", raising=False)
    monkeypatch.delenv("VPI_EMBEDDING_MAX_LENGTH", raising=False)

    configuration = load_vpi_embedding_provider_execution_configuration()

    assert configuration.device is None
    assert configuration.provider_batch_size is None


def test_execution_env_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPI_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("VPI_EMBEDDING_PROVIDER_BATCH_SIZE", "64")
    monkeypatch.setenv("VPI_EMBEDDING_MAX_LENGTH", "512")

    configuration = load_vpi_embedding_provider_execution_configuration()

    assert configuration.device == "cuda"
    assert configuration.provider_batch_size == 64


def test_explicit_cuda_unavailable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    configuration = VpiEmbeddingProviderExecutionConfiguration(
        execution=EmbeddingProviderExecutionConfig(device="cuda")
    )
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: False,
    )

    with pytest.raises(VpiEmbeddingDeviceUnavailableError, match="CUDA is unavailable"):
        assert_execution_device_available(configuration)
