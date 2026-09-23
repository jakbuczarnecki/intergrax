# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1 — canonical provider identity normalization."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider, llm_provider_slug


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (LLMProvider.OPENAI, "openai"),
        ("openai", "openai"),
        (" OpenAI ", "openai"),
        ("external-provider", "external-provider"),
        (" External-Provider ", "external-provider"),
    ],
)
def test_llm_provider_slug_normalization_matrix(
    provider: LLMProvider | str, expected: str
) -> None:
    assert llm_provider_slug(provider) == expected


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("provider", ["", "   "])
def test_llm_provider_slug_rejects_empty(provider: str) -> None:
    with pytest.raises(ValueError, match="provider must not be empty"):
        llm_provider_slug(provider)
