# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.registrations._lazy_factory import register_lazy_adapter
from intergrax.llm_adapters.registry.registration_contract import OptionalDependencyRequirement

_OPENAI_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("openai",),
    distribution_name="openai",
    extra_name="llm-openai",
)


def _load_openai_responses_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.openai_responses_adapter import (
        OpenAIChatResponsesAdapter,
    )

    return OpenAIChatResponsesAdapter


def _load_azure_openai_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.azure_openai_adapter import AzureOpenAIChatAdapter

    return AzureOpenAIChatAdapter


def register_openai(registry: type) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.OPENAI.value,
        dependency=_OPENAI_DEPENDENCY,
        load_adapter_cls=_load_openai_responses_adapter,
    )


def register_azure_openai(registry: type) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.AZURE_OPENAI.value,
        dependency=_OPENAI_DEPENDENCY,
        load_adapter_cls=_load_azure_openai_adapter,
    )
