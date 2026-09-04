"""Scenario-owned embedding provider execution tuning — not artifact identity."""

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

VPI_EMBEDDING_EXECUTION_ENV_PREFIX = "VPI_EMBEDDING"


class VpiEmbeddingDeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested execution device is unavailable."""


@dataclass(frozen=True, slots=True)
class VpiEmbeddingProviderExecutionConfiguration:
    """VPI operator execution settings for the reference embedding provider."""

    execution: EmbeddingProviderExecutionConfig

    @property
    def device(self) -> str | None:
        return self.execution.device

    @property
    def provider_batch_size(self) -> int | None:
        return self.execution.batch_size


def _parse_optional_positive_int(raw_value: str | None) -> int | None:
    if raw_value is None or not raw_value.strip():
        return None
    parsed = int(raw_value.strip())
    if parsed <= 0:
        msg = "batch size must be a positive integer"
        raise ValueError(msg)
    return parsed


def _parse_optional_device(raw_value: str | None) -> str | None:
    if raw_value is None or not raw_value.strip():
        return None
    return raw_value.strip()


def load_vpi_embedding_provider_execution_configuration(
    *,
    prefix: str = VPI_EMBEDDING_EXECUTION_ENV_PREFIX,
) -> VpiEmbeddingProviderExecutionConfiguration:
    device = _parse_optional_device(os.getenv(f"{prefix}_DEVICE"))
    provider_batch_size = _parse_optional_positive_int(
        os.getenv(f"{prefix}_PROVIDER_BATCH_SIZE")
    )
    execution = EmbeddingProviderExecutionConfig(
        device=device,
        batch_size=provider_batch_size,
    )
    return VpiEmbeddingProviderExecutionConfiguration(execution=execution)


def assert_execution_device_available(
    configuration: VpiEmbeddingProviderExecutionConfiguration,
) -> None:
    """Fail closed when CUDA is explicitly requested but unavailable."""
    requested = configuration.device
    if requested is None:
        return
    normalized = requested.strip().casefold()
    if normalized != "cuda":
        return
    try:
        import torch
    except ImportError as exc:
        raise VpiEmbeddingDeviceUnavailableError(
            "VPI_EMBEDDING_DEVICE=cuda requested but torch is not installed"
        ) from exc
    if not torch.cuda.is_available():
        raise VpiEmbeddingDeviceUnavailableError(
            "VPI_EMBEDDING_DEVICE=cuda requested but CUDA is unavailable in the current torch build"
        )
