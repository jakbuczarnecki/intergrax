# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.registrations._lazy_factory import register_lazy_adapter
from intergrax.llm_adapters.providers.registrations.external_operation_seams import (
    HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES,
    OLLAMA_LOCAL_EXTERNAL_OPERATION_CAPABILITIES,
    build_ollama_external_operation_seam,
    build_openai_external_operation_seam,
)
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterRegistrationTarget,
    OptionalDependencyRequirement,
)

_OPENAI_COMPAT_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("openai",),
    distribution_name="openai",
    extra_name="llm-compat",
)

_GROQ_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("openai",),
    distribution_name="openai",
    extra_name="llm-groq",
)

_VLLM_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("openai",),
    distribution_name="openai",
    extra_name="llm-vllm",
)


def _load_groq_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import GroqChatAdapter

    return GroqChatAdapter


def _load_vllm_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import VllmChatAdapter

    return VllmChatAdapter


def _load_together_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import TogetherChatAdapter

    return TogetherChatAdapter


def _load_fireworks_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import FireworksChatAdapter

    return FireworksChatAdapter


def _load_openrouter_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import OpenRouterChatAdapter

    return OpenRouterChatAdapter


def _load_deepseek_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import DeepSeekChatAdapter

    return DeepSeekChatAdapter


def _load_xai_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import XaiChatAdapter

    return XaiChatAdapter


def _load_llama_cpp_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import LlamaCppChatAdapter

    return LlamaCppChatAdapter


def _load_cohere_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import CohereChatAdapter

    return CohereChatAdapter


def _load_azure_ai_inference_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_compat_providers import (
        AzureAiInferenceChatAdapter,
    )

    return AzureAiInferenceChatAdapter


def _register_openai_compat_adapter(
    registry: LLMAdapterRegistrationTarget,
    *,
    provider_id: str,
    dependency: OptionalDependencyRequirement | None,
    load_adapter_cls: Callable[[], type[LLMAdapter]],
) -> None:
    register_lazy_adapter(
        registry,
        provider_id=provider_id,
        dependency=dependency,
        load_adapter_cls=load_adapter_cls,
        external_operation_seam_factory=build_openai_external_operation_seam,
        external_operation_capabilities=HTTP_SDK_EXTERNAL_OPERATION_CAPABILITIES,
    )


def register_groq(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.GROQ.value,
        dependency=_GROQ_DEPENDENCY,
        load_adapter_cls=_load_groq_adapter,
    )


def register_vllm(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.VLLM.value,
        dependency=_VLLM_DEPENDENCY,
        load_adapter_cls=_load_vllm_adapter,
    )


def register_together(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.TOGETHER.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_together_adapter,
    )


def register_fireworks(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.FIREWORKS.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_fireworks_adapter,
    )


def register_openrouter(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.OPENROUTER.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_openrouter_adapter,
    )


def register_deepseek(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.DEEPSEEK.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_deepseek_adapter,
    )


def register_xai(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.XAI.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_xai_adapter,
    )


def register_llama_cpp(registry: LLMAdapterRegistrationTarget) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.LLAMA_CPP.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_llama_cpp_adapter,
        external_operation_seam_factory=build_ollama_external_operation_seam,
        external_operation_capabilities=OLLAMA_LOCAL_EXTERNAL_OPERATION_CAPABILITIES,
    )


def register_cohere(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.COHERE.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_cohere_adapter,
    )


def register_azure_ai_inference(registry: LLMAdapterRegistrationTarget) -> None:
    _register_openai_compat_adapter(
        registry,
        provider_id=LLMProvider.AZURE_AI_INFERENCE.value,
        dependency=_OPENAI_COMPAT_DEPENDENCY,
        load_adapter_cls=_load_azure_ai_inference_adapter,
    )
