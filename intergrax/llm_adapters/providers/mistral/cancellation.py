# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mistral SDK stream termination (W4-D)."""

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


class MistralNativeTermination(StreamRegistryTerminationPort):
    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        result = await StreamRegistryTerminationPort.terminate(self, identity)
        if result.outcome.value == "transport_closed":
            return TerminationResult.physical_stop_confirmed()
        return result


def mistral_external_operation_ports() -> tuple[
    RegistryCancellationPort,
    MistralNativeTermination,
    ProviderStreamTransportRegistry,
]:
    registry = ProviderStreamTransportRegistry()
    termination = MistralNativeTermination(registry)
    return RegistryCancellationPort(registry), termination, registry
