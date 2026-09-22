# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral external-operation defaults when no seam is registered."""

from __future__ import annotations

from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    TerminationResult,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters.registry.registration_contract import (
    ProviderExternalOperationSeam,
)


class NoOpExternalOperationCancellationPort:
    """Providers without cancel API — intent-only cancellation."""

    async def request_cancel(self, operation_id: str) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")


class NoOpExternalOperationTerminationPort:
    async def terminate(
        self,
        identity: ExternalOperationIdentity,
    ) -> TerminationResult:
        return TerminationResult.not_supported()


_UNREGISTERED_CAPABILITIES = ExternalOperationCapabilities(
    supports_native_cancel=False,
    supports_stream_abort=False,
    supports_remote_termination=False,
)


def default_external_operation_seam() -> ProviderExternalOperationSeam:
    return ProviderExternalOperationSeam(
        cancellation=NoOpExternalOperationCancellationPort(),
        termination=NoOpExternalOperationTerminationPort(),
        stream_registry=ProviderStreamTransportRegistry(),
        capabilities=_UNREGISTERED_CAPABILITIES,
    )
