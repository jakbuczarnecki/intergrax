# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Replicate ml inference host integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import MlInferenceHostIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID = "replicate"


class ReplicateMlInferenceHostIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Replicate ml inference host integration."""

    pass


@runtime_checkable
class ReplicateMlInferenceHostClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ReplicateMlInferenceHostIntegration(MlInferenceHostIntegrationContract):
    """
    Replicate ml inference host integration.

    The legacy facade (create_replicate_ml_inference_host) remains separate and backward-compatible.
    """

    config: ReplicateMlInferenceHostIntegrationConfig = ReplicateMlInferenceHostIntegrationConfig()
    _client: ReplicateMlInferenceHostClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ReplicateMlInferenceHostClient,
        *,
        enabled: bool = False,
    ) -> ReplicateMlInferenceHostIntegration:
        integration = cls.for_provider(
            provider_id=REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID,
            display_name="Replicate",
            config=ReplicateMlInferenceHostIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ReplicateMlInferenceHostClient | None:
        return self._client
