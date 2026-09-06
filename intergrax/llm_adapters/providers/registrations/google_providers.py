# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.registrations._lazy_factory import register_lazy_adapter
from intergrax.llm_adapters.registry.registration_contract import OptionalDependencyRequirement

_GEMINI_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("google", "google.genai"),
    distribution_name="google-genai",
    extra_name="llm-gemini",
)

_VERTEX_GEMINI_DEPENDENCY = OptionalDependencyRequirement(
    import_names=("google", "google.genai"),
    distribution_name="google-genai",
    extra_name="llm-vertex",
)


def _load_gemini_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter

    return GeminiChatAdapter


def _load_vertex_gemini_adapter() -> type[LLMAdapter]:
    from intergrax.llm_adapters.providers.vertex_gemini_adapter import VertexGeminiChatAdapter

    return VertexGeminiChatAdapter


def register_gemini(registry: type) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.GEMINI.value,
        dependency=_GEMINI_DEPENDENCY,
        load_adapter_cls=_load_gemini_adapter,
    )


def register_vertex_gemini(registry: type) -> None:
    register_lazy_adapter(
        registry,
        provider_id=LLMProvider.VERTEX_GEMINI.value,
        dependency=_VERTEX_GEMINI_DEPENDENCY,
        load_adapter_cls=_load_vertex_gemini_adapter,
    )
