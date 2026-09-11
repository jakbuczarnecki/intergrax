# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLM provider cancellation bridge (W4-C) — best-effort, no central manager."""

from __future__ import annotations

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationCancellationPort,
    ExternalOperationPhysicalState,
    ExternalOperationStatusPort,
)


class NoOpExternalOperationCancellationPort:
    """Providers without cancel API — intent-only cancellation."""

    async def request_cancel(self, operation_id: str) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")


class UnknownOnInquiryExternalOperationStatusPort:
    """Status inquiry when provider has no durable remote status."""

    async def get_status(self, operation_id: str) -> ExternalOperationPhysicalState:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        return ExternalOperationPhysicalState.UNKNOWN


def bind_llm_external_operation_ports(
    adapter: object,
    *,
    cancellation_port: ExternalOperationCancellationPort | None,
    status_port: ExternalOperationStatusPort | None,
) -> None:
    """Attach optional W4-C ports on LLMAdapter instances."""
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

    if not isinstance(adapter, LLMAdapter):
        raise TypeError("adapter must be LLMAdapter")
    adapter.bind_external_operation_ports(
        store=None,
        cancellation_port=cancellation_port,
        status_port=status_port,
    )
