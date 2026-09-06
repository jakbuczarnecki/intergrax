# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.registrations._lazy_factory import register_lazy_adapter
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterRegistrationTarget,
    OptionalDependencyRequirement,
)

_CLAUDE_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("anthropic",),
    distribution_name="anthropic",
    extra_name="llm-anthropic",
)

_MISTRAL_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("mistralai",),
    distribution_name="mistralai",
    extra_name="llm-mistral",
)

_BEDROCK_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("boto3",),
    distribution_name="boto3",
    extra_name="llm-bedrock",
)

_OLLAMA_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("ollama",),
    distribution_name="ollama",
    extra_name="llm-ollama",
)

_COHERE_NATIVE_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("cohere",),
    distribution_name="cohere",
    extra_name="llm-cohere-native",
)


def _load_claude_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    return ClaudeChatAdapter


def _load_mistral_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.mistral_adapter import MistralChatAdapter

    return MistralChatAdapter


def _load_bedrock_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

    return BedrockChatAdapter


def _load_native_ollama_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter

    return NativeOllamaAdapter


def _load_cohere_native_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.cohere_native_adapter import CohereNativeChatAdapter

    return CohereNativeChatAdapter


def register_claude(registry: LLMAdapterRegistrationTarget) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.CLAUDE.value,
        dependency=_CLAUDE_DEPENDENCY,
        load_adapter_cls=_load_claude_adapter,
    )


def register_mistral(registry: LLMAdapterRegistrationTarget) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.MISTRAL.value,
        dependency=_MISTRAL_DEPENDENCY,
        load_adapter_cls=_load_mistral_adapter,
    )


def register_aws_bedrock(registry: LLMAdapterRegistrationTarget) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.AWS_BEDROCK.value,
        dependency=_BEDROCK_DEPENDENCY,
        load_adapter_cls=_load_bedrock_adapter,
    )


def register_ollama(registry: LLMAdapterRegistrationTarget) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.OLLAMA.value,
        dependency=_OLLAMA_DEPENDENCY,
        load_adapter_cls=_load_native_ollama_adapter,
    )


def register_cohere_native(registry: LLMAdapterRegistrationTarget) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.COHERE_NATIVE.value,
        dependency=_COHERE_NATIVE_DEPENDENCY,
        load_adapter_cls=_load_cohere_native_adapter,
    )
