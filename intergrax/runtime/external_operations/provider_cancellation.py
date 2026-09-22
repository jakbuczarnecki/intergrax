# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLM provider cancellation bridge (W4-C/D) — best-effort, no central manager."""

from __future__ import annotations

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationCancellationPort,
    ExternalOperationPhysicalState,
    ExternalOperationStatusPort,
)
from intergrax.contracts.external_operation_termination import (
    ExternalOperationTerminationPort,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters.base.lifecycle_binding import LLMRuntimeLifecycleBinding
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class UnknownOnInquiryExternalOperationStatusPort:
    """Status inquiry when provider has no durable remote status."""

    async def get_status(self, operation_id: str) -> ExternalOperationPhysicalState:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        return ExternalOperationPhysicalState.UNKNOWN


def resolve_llm_provider_external_operation_seam(
    provider_slug: str,
) -> tuple[
    ExternalOperationCancellationPort,
    ExternalOperationTerminationPort,
    ProviderStreamTransportRegistry,
]:
    """Resolve provider-owned external-operation ports via adapter registration."""
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

    seam = LLMAdapterRegistry.resolve_external_operation_seam(provider_slug)
    return seam.cancellation, seam.termination, seam.stream_registry


def bind_llm_external_operation_ports(
    adapter: LLMAdapter,
    *,
    cancellation_port: ExternalOperationCancellationPort | None,
    status_port: ExternalOperationStatusPort | None,
    termination_port: ExternalOperationTerminationPort | None = None,
    stream_registry: ProviderStreamTransportRegistry | None = None,
) -> None:
    """Attach optional W4-C/D ports on LLMAdapter instances."""
    if not isinstance(adapter, LLMRuntimeLifecycleBinding):
        return
    adapter.bind_external_operation_ports(
        store=None,
        cancellation_port=cancellation_port,
        status_port=status_port,
        termination_port=termination_port,
        stream_registry=stream_registry,
    )
