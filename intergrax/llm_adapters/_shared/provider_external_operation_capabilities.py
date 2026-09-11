# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability discovery for LLM provider cancellation (W4-D)."""

from __future__ import annotations

from intergrax.contracts.external_operation_termination import ExternalOperationCapabilities
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

_HTTP_SDK_REMOTE = ExternalOperationCapabilities(
    supports_native_cancel=True,
    supports_stream_abort=True,
    supports_remote_termination=True,
)

_OLLAMA_LOCAL = ExternalOperationCapabilities(
    supports_native_cancel=False,
    supports_stream_abort=True,
    supports_remote_termination=False,
)

_OPENAI_FAMILY = ExternalOperationCapabilities(
    supports_native_cancel=True,
    supports_stream_abort=True,
    supports_remote_termination=True,
)

_CAPABILITIES_BY_SLUG: dict[str, ExternalOperationCapabilities] = {
    LLMProvider.OPENAI.value: _OPENAI_FAMILY,
    LLMProvider.AZURE_OPENAI.value: _OPENAI_FAMILY,
    LLMProvider.CLAUDE.value: _HTTP_SDK_REMOTE,
    LLMProvider.GEMINI.value: _HTTP_SDK_REMOTE,
    LLMProvider.VERTEX_GEMINI.value: _HTTP_SDK_REMOTE,
    LLMProvider.MISTRAL.value: _HTTP_SDK_REMOTE,
    LLMProvider.AWS_BEDROCK.value: ExternalOperationCapabilities(
        supports_native_cancel=False,
        supports_stream_abort=True,
        supports_remote_termination=True,
    ),
    LLMProvider.OLLAMA.value: _OLLAMA_LOCAL,
    LLMProvider.GROQ.value: _HTTP_SDK_REMOTE,
    LLMProvider.VLLM.value: _HTTP_SDK_REMOTE,
    LLMProvider.TOGETHER.value: _HTTP_SDK_REMOTE,
    LLMProvider.FIREWORKS.value: _HTTP_SDK_REMOTE,
    LLMProvider.OPENROUTER.value: _HTTP_SDK_REMOTE,
    LLMProvider.DEEPSEEK.value: _HTTP_SDK_REMOTE,
    LLMProvider.XAI.value: _HTTP_SDK_REMOTE,
    LLMProvider.LLAMA_CPP.value: _OLLAMA_LOCAL,
    LLMProvider.COHERE.value: _HTTP_SDK_REMOTE,
    LLMProvider.COHERE_NATIVE.value: _HTTP_SDK_REMOTE,
    LLMProvider.AZURE_AI_INFERENCE.value: _HTTP_SDK_REMOTE,
}


def external_operation_capabilities_for_provider(
    provider_slug: str,
) -> ExternalOperationCapabilities:
    normalized = provider_slug.strip().lower()
    return _CAPABILITIES_BY_SLUG.get(
        normalized,
        ExternalOperationCapabilities(
            supports_native_cancel=False,
            supports_stream_abort=False,
            supports_remote_termination=False,
        ),
    )
