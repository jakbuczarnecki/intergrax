# © Artur Czarnecki. All rights reserved.

"""Unit tests for DS-E2E provider independence contracts."""

from __future__ import annotations

import pytest

from testing_support.decision_e2e.bindings import ProviderBindingEvidence
from testing_support.decision_e2e.independence import (
    ProviderIndependenceLevel,
    council_requires_distinct_models,
    evaluate_provider_independence,
    producer_verifier_requires_distinct_models,
)


def _binding(profile_id: str, provider: str, model: str) -> ProviderBindingEvidence:
    return ProviderBindingEvidence(profile_id=profile_id, provider=provider, model=model)


def test_profile_only_same_model_different_profiles_blocked_for_council() -> None:
    bindings = (
        _binding("profile-a", "ollama", "llama3.1:8b"),
        _binding("profile-b", "ollama", "llama3.1:8b"),
        _binding("profile-c", "ollama", "llama3.1:8b"),
    )
    result = evaluate_provider_independence(bindings)
    assert result.level is ProviderIndependenceLevel.PROFILE_ONLY
    qualifies, reason = council_requires_distinct_models(bindings)
    assert qualifies is False
    assert reason is not None
    assert "distinct real model identities" in reason


def test_same_provider_different_models_eligible_for_council() -> None:
    bindings = (
        _binding("profile-a", "ollama", "llama3.1:8b"),
        _binding("profile-b", "ollama", "qwen3:8b"),
        _binding("profile-c", "ollama", "llama3.1:8b"),
    )
    result = evaluate_provider_independence(bindings)
    assert result.level is ProviderIndependenceLevel.DISTINCT_MODEL
    qualifies, reason = council_requires_distinct_models(bindings)
    assert qualifies is True
    assert reason is None


def test_different_providers_same_model_string_is_provider_independent() -> None:
    bindings = (
        _binding("profile-a", "ollama", "llama3.1:8b"),
        _binding("profile-b", "openai", "llama3.1:8b"),
    )
    result = evaluate_provider_independence(bindings)
    assert result.level in {
        ProviderIndependenceLevel.DISTINCT_PROVIDER,
        ProviderIndependenceLevel.DISTINCT_PROVIDER_AND_MODEL,
    }
    qualifies, _ = council_requires_distinct_models(bindings)
    assert qualifies is True


def test_producer_verifier_same_model_blocked() -> None:
    producer = _binding("profile-producer", "ollama", "llama3.1:8b")
    verifier = _binding("profile-verifier", "ollama", "llama3.1:8b")
    qualifies, reason = producer_verifier_requires_distinct_models(producer, verifier)
    assert qualifies is False
    assert reason is not None


def test_producer_verifier_different_model_eligible() -> None:
    producer = _binding("profile-producer", "ollama", "llama3.1:8b")
    verifier = _binding("profile-verifier", "ollama", "qwen3:8b")
    qualifies, reason = producer_verifier_requires_distinct_models(producer, verifier)
    assert qualifies is True
    assert reason is None
