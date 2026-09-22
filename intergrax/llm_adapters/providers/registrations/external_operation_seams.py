# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-owned external-operation seam factories for adapter registration."""

from __future__ import annotations

from intergrax.contracts.external_operation_termination import ExternalOperationCapabilities
from intergrax.llm_adapters.registry.registration_contract import ProviderExternalOperationSeam

HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES = ExternalOperationCapabilities(
    supports_native_cancel=True,
    supports_stream_abort=True,
    supports_remote_termination=True,
)

OPENAI_FAMILY_EXTERNAL_OPERATION_CAPABILITIES = HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES

OLLAMA_LOCAL_EXTERNAL_OPERATION_CAPABILITIES = ExternalOperationCapabilities(
    supports_native_cancel=False,
    supports_stream_abort=True,
    supports_remote_termination=False,
)

BEDROCK_EXTERNAL_OPERATION_CAPABILITIES = ExternalOperationCapabilities(
    supports_native_cancel=False,
    supports_stream_abort=True,
    supports_remote_termination=True,
)


def build_openai_external_operation_seam() -> ProviderExternalOperationSeam:
    from intergrax.llm_adapters.providers.openai.cancellation import (
        openai_external_operation_ports,
    )

    cancel, termination, registry = openai_external_operation_ports()
    return ProviderExternalOperationSeam(
        cancellation=cancel,
        termination=termination,
        stream_registry=registry,
    )


def build_claude_external_operation_seam() -> ProviderExternalOperationSeam:
    from intergrax.llm_adapters.providers.claude.cancellation import (
        claude_external_operation_ports,
    )

    cancel, termination, registry = claude_external_operation_ports()
    return ProviderExternalOperationSeam(
        cancellation=cancel,
        termination=termination,
        stream_registry=registry,
    )


def build_gemini_external_operation_seam() -> ProviderExternalOperationSeam:
    from intergrax.llm_adapters.providers.gemini.cancellation import (
        gemini_external_operation_ports,
    )

    cancel, termination, registry = gemini_external_operation_ports()
    return ProviderExternalOperationSeam(
        cancellation=cancel,
        termination=termination,
        stream_registry=registry,
    )


def build_mistral_external_operation_seam() -> ProviderExternalOperationSeam:
    from intergrax.llm_adapters.providers.mistral.cancellation import (
        mistral_external_operation_ports,
    )

    cancel, termination, registry = mistral_external_operation_ports()
    return ProviderExternalOperationSeam(
        cancellation=cancel,
        termination=termination,
        stream_registry=registry,
    )


def build_bedrock_external_operation_seam() -> ProviderExternalOperationSeam:
    from intergrax.llm_adapters.providers.aws_bedrock.cancellation import (
        bedrock_external_operation_ports,
    )

    cancel, termination, registry = bedrock_external_operation_ports()
    return ProviderExternalOperationSeam(
        cancellation=cancel,
        termination=termination,
        stream_registry=registry,
    )


def build_ollama_external_operation_seam() -> ProviderExternalOperationSeam:
    from intergrax.llm_adapters.providers.ollama.cancellation import (
        ollama_external_operation_ports,
    )

    cancel, termination, registry = ollama_external_operation_ports()
    return ProviderExternalOperationSeam(
        cancellation=cancel,
        termination=termination,
        stream_registry=registry,
    )
