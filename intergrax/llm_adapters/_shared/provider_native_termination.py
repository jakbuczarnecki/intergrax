# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared HTTP/SDK stream termination seam (W4-D)."""

from __future__ import annotations

from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import (
    TerminationResult,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)


class StreamRegistryTerminationPort:
    """Close in-flight stream transport registered for operation_id."""

    __slots__ = ("_registry",)

    def __init__(self, registry: ProviderStreamTransportRegistry) -> None:
        self._registry = registry

    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        closed = self._registry.close_transport(identity.operation_id)
        if closed:
            return TerminationResult.transport_closed()
        return TerminationResult.not_supported()
