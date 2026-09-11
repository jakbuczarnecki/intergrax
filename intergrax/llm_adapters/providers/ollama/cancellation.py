# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ollama local process — no native remote cancel (W4-D)."""

from __future__ import annotations

from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import TerminationResult
from intergrax.llm_adapters._shared.provider_native_termination import (
    StreamRegistryTerminationPort,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters._shared.registry_cancellation_port import (
    RegistryCancellationPort,
)


class OllamaNativeTermination(StreamRegistryTerminationPort):
    """Local HTTP stream close only; no process kill."""

    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        result = await StreamRegistryTerminationPort.terminate(self, identity)
        if result.outcome.value == "transport_closed":
            return TerminationResult.transport_closed()
        return TerminationResult.not_supported()


def ollama_external_operation_ports() -> tuple[
    RegistryCancellationPort,
    OllamaNativeTermination,
    ProviderStreamTransportRegistry,
]:
    registry = ProviderStreamTransportRegistry()
    termination = OllamaNativeTermination(registry)
    return RegistryCancellationPort(registry), termination, registry
