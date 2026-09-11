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
from intergrax.llm_adapters._shared.provider_external_operation_capabilities import (
    external_operation_capabilities_for_provider,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


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


def resolve_llm_provider_external_operation_seam(
    provider_slug: str,
) -> tuple[
    ExternalOperationCancellationPort,
    ExternalOperationTerminationPort,
    ProviderStreamTransportRegistry,
]:
    """Capability-driven seam wiring — no provider string branching in runtime core."""
    normalized = provider_slug.strip().lower()
    if normalized in (
        LLMProvider.OPENAI.value,
        LLMProvider.AZURE_OPENAI.value,
        LLMProvider.GROQ.value,
        LLMProvider.VLLM.value,
        LLMProvider.OPENROUTER.value,
    ):
        from intergrax.llm_adapters.providers.openai.cancellation import (
            openai_external_operation_ports,
        )

        cancel, termination, registry = openai_external_operation_ports()
        return cancel, termination, registry
    if normalized == LLMProvider.CLAUDE.value:
        from intergrax.llm_adapters.providers.claude.cancellation import (
            claude_external_operation_ports,
        )

        cancel, termination, registry = claude_external_operation_ports()
        return cancel, termination, registry
    if normalized in (LLMProvider.GEMINI.value, LLMProvider.VERTEX_GEMINI.value):
        from intergrax.llm_adapters.providers.gemini.cancellation import (
            gemini_external_operation_ports,
        )

        cancel, termination, registry = gemini_external_operation_ports()
        return cancel, termination, registry
    if normalized == LLMProvider.MISTRAL.value:
        from intergrax.llm_adapters.providers.mistral.cancellation import (
            mistral_external_operation_ports,
        )

        cancel, termination, registry = mistral_external_operation_ports()
        return cancel, termination, registry
    if normalized == LLMProvider.AWS_BEDROCK.value:
        from intergrax.llm_adapters.providers.aws_bedrock.cancellation import (
            bedrock_external_operation_ports,
        )

        cancel, termination, registry = bedrock_external_operation_ports()
        return cancel, termination, registry
    if normalized == LLMProvider.OLLAMA.value:
        from intergrax.llm_adapters.providers.ollama.cancellation import (
            ollama_external_operation_ports,
        )

        cancel, termination, registry = ollama_external_operation_ports()
        return cancel, termination, registry
    caps = external_operation_capabilities_for_provider(normalized)
    if caps.supports_native_cancel or caps.supports_stream_abort:
        from intergrax.llm_adapters.providers.openai.cancellation import (
            openai_external_operation_ports,
        )

        cancel, termination, registry = openai_external_operation_ports()
        return cancel, termination, registry
    return (
        NoOpExternalOperationCancellationPort(),
        _NoOpTerminationPort(),
        ProviderStreamTransportRegistry(),
    )


class _NoOpTerminationPort:
    async def terminate(
        self,
        identity: object,
    ) -> object:
        from intergrax.contracts.external_operation_identity import (
            ExternalOperationIdentity,
        )
        from intergrax.contracts.external_operation_termination import TerminationResult

        if not isinstance(identity, ExternalOperationIdentity):
            raise TypeError("identity must be ExternalOperationIdentity")
        return TerminationResult.not_supported()


def bind_llm_external_operation_ports(
    adapter: object,
    *,
    cancellation_port: ExternalOperationCancellationPort | None,
    status_port: ExternalOperationStatusPort | None,
    termination_port: ExternalOperationTerminationPort | None = None,
    stream_registry: ProviderStreamTransportRegistry | None = None,
) -> None:
    """Attach optional W4-C/D ports on LLMAdapter instances."""
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

    if not isinstance(adapter, LLMAdapter):
        raise TypeError("adapter must be LLMAdapter")
    adapter.bind_external_operation_ports(
        store=None,
        cancellation_port=cancellation_port,
        status_port=status_port,
        termination_port=termination_port,
        stream_registry=stream_registry,
    )
