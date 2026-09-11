# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cancellation port that closes adapter-local stream transports (W4-D)."""

from __future__ import annotations

from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)


class RegistryCancellationPort:
    """ExternalOperationCancellationPort via stream transport registry."""

    __slots__ = ("_registry",)

    def __init__(self, registry: ProviderStreamTransportRegistry) -> None:
        self._registry = registry

    async def request_cancel(self, operation_id: str) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        self._registry.close_transport(operation_id)
