# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Self-hosted model serving runtime contract (local/remote inference hosts)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import HealthStatus


@runtime_checkable
class ModelServingRuntimeBackend(Protocol):
    """
    Self-hosted model serving host (Ollama, vLLM, llama.cpp server, TGI, …).

    Distinct from ``MlInferenceHostBackend`` (managed remote ``predict`` endpoints).
    """

    def list_models(self) -> list[str]:
        """Return model identifiers available on the serving host."""

    def health(self) -> HealthStatus | bool:
        """Probe whether the serving host is reachable."""


__all__ = ["ModelServingRuntimeBackend"]
