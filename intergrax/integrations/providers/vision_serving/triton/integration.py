# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Triton vision serving integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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
    Triton vision serving integration.

    The legacy facade (create_triton_vision_serving) remains separate and backward-compatible.
    """

    config: TritonVisionServingIntegrationConfig = TritonVisionServingIntegrationConfig()
    _client: TritonVisionServingClient | None = PrivateAttr(default=None)

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
