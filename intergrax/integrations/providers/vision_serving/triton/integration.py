# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Triton vision serving integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vision_serving import VisionServingBackend
from intergrax.runtime.integrations.categories.ai import VisionServingIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TRITON_VISION_SERVING_PROVIDER_ID = "triton"


class TritonVisionServingIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Triton vision serving integration."""

    pass


TritonVisionServingClient = VisionServingBackend

class TritonVisionServingIntegration(VisionServingIntegrationContract):
    """
    Single public Triton vision serving entrypoint.

    Legacy catalog factory (create_triton_vision_serving) owns catalog behavior; legacy factories use from_client().
    """

    config: TritonVisionServingIntegrationConfig = TritonVisionServingIntegrationConfig()
    _client: TritonVisionServingClient | None = PrivateAttr(default=None)
    

    def predict(self, model_name, input_uri):
        return self._require_client().predict(model_name, input_uri)

    def _require_client(self) -> VisionServingBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

VisionServingBackend.register(TritonVisionServingIntegration)
