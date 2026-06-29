# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Triton vision serving integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vision_serving import VisionServingBackend
from intergrax.runtime.integrations.categories.ai import VisionServingIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TRITON_VISION_SERVING_PROVIDER_ID = "triton"


class TritonVisionServingIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Triton vision serving integration."""

    pass


@runtime_checkable
class TritonVisionServingClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TritonVisionServingIntegration(VisionServingIntegrationContract):
    """
    Single public Triton vision serving entrypoint.

    Legacy catalog factory (create_triton_vision_serving) delegates to this class.
    """

    config: TritonVisionServingIntegrationConfig = TritonVisionServingIntegrationConfig()
    _client: TritonVisionServingClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> TritonVisionServingIntegration:
        integration = cls.for_provider(
            provider_id=TRITON_VISION_SERVING_PROVIDER_ID,
            display_name="Triton",
            config=TritonVisionServingIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Triton integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: TritonVisionServingClient,
        *,
        enabled: bool = False,
    ) -> TritonVisionServingIntegration:
        integration = cls.for_provider(
            provider_id=TRITON_VISION_SERVING_PROVIDER_ID,
            display_name="Triton",
            config=TritonVisionServingIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TritonVisionServingClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

VisionServingBackend.register(TritonVisionServingIntegration)
