# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.model_router import ModelRouter
from intergrax.llm_adapters.registry.profile import LLMProfile


def _profile(provider: LLMProvider, model: str) -> LLMProfile:
    return LLMProfile(provider=provider, model=model)


@pytest.mark.unit
@pytest.mark.gate
def test_model_router_balanced_prefers_first_fallback() -> None:
    primary = _profile(LLMProvider.OPENAI, "gpt-4o")
    fallback = _profile(LLMProvider.GROQ, "llama-3.3-70b-versatile")
    router = ModelRouter.from_profiles(
        primary,
        fallbacks=(fallback,),
        policy_route_hint="balanced",
    )
    ordered = router.ordered_profiles()
    assert ordered[0].model == "llama-3.3-70b-versatile"
    decision = router.resolve()
    assert decision.routing_reason == "policy_hint_balanced"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("hint", "expected_first_model"),
    [
        ("cheapest", "cheap-model"),
        ("fastest", "fast-model"),
        ("quality", "fast-model"),
    ],
)
def test_model_router_hint_ordering(hint: str, expected_first_model: str) -> None:
    primary = _profile(LLMProvider.OPENAI, "fast-model")
    cheap = _profile(LLMProvider.GROQ, "cheap-model")
    router = ModelRouter.from_profiles(
        primary,
        fallbacks=(cheap,),
        policy_route_hint=hint,
    )
    assert router.ordered_profiles()[0].model == expected_first_model
