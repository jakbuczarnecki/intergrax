# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Replicate ml inference host integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ml_inference_host import MlInferenceHostBackend
from intergrax.runtime.integrations.categories.ai import MlInferenceHostIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID = "replicate"


class ReplicateMlInferenceHostIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Replicate ml inference host integration."""

    pass


ReplicateMlInferenceHostClient = MlInferenceHostBackend

class ReplicateMlInferenceHostIntegration(MlInferenceHostIntegrationContract):
    """
    Single public Replicate ml inference host entrypoint.

    Legacy catalog factory (create_replicate_ml_inference_host) owns catalog behavior; legacy factories use from_client().
    """

    config: ReplicateMlInferenceHostIntegrationConfig = ReplicateMlInferenceHostIntegrationConfig()
    _client: ReplicateMlInferenceHostClient | None = PrivateAttr(default=None)
    

    def predict(self, model_ref, inputs):
        return self._require_client().predict(model_ref, inputs)

    def _require_client(self) -> MlInferenceHostBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

MlInferenceHostBackend.register(ReplicateMlInferenceHostIntegration)
