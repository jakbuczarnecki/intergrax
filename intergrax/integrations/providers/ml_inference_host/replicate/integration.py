# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Replicate ml inference host integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ml_inference_host import MlInferenceHostBackend
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
    Single public Replicate ml inference host entrypoint.

    Legacy catalog factory (create_replicate_ml_inference_host) delegates to this class.
    """

    config: ReplicateMlInferenceHostIntegrationConfig = ReplicateMlInferenceHostIntegrationConfig()
    _client: ReplicateMlInferenceHostClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> ReplicateMlInferenceHostIntegration:
        integration = cls.for_provider(
            provider_id=REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID,
            display_name="Replicate",
            config=ReplicateMlInferenceHostIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Replicate integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

MlInferenceHostBackend.register(ReplicateMlInferenceHostIntegration)
