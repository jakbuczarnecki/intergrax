# © Artur Czarnecki. All rights reserved.

"""Deterministic LLM adapters for EBH-7 factory RPL qualification (no network)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_provider import llm_provider_slug
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.registration_contract import LLMAdapterRegistrationSpec


class _QualAlphaLLMAdapter(BaseLLMAdapter):
    provider = "ebh7_qual_alpha"
    model = "ebh7-qual-alpha"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id, messages
        return build_adapter_response(content="ebh7-alpha")


class _QualBetaLLMAdapter(BaseLLMAdapter):
    provider = "ebh7_qual_beta"
    model = "ebh7-qual-beta"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id, messages
        return build_adapter_response(content="ebh7-beta")


def reset_and_register_qualification_llm_providers() -> tuple[str, str]:
    LLMAdapterRegistry.reset_for_testing()
    alpha = llm_provider_slug("ebh7_qual_alpha")
    beta = llm_provider_slug("ebh7_qual_beta")
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(provider_id=alpha, factory=lambda **_k: _QualAlphaLLMAdapter()),
        override=True,
    )
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(provider_id=beta, factory=lambda **_k: _QualBetaLLMAdapter()),
        override=True,
    )
    return alpha, beta
