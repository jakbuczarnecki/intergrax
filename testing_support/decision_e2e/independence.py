# © Artur Czarnecki. All rights reserved.

"""Typed provider independence contracts for DS-E2E qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from testing_support.decision_e2e.bindings import ProviderBindingEvidence


class ProviderIndependenceLevel(StrEnum):
    """Machine-verifiable independence classification."""

    PROFILE_ONLY = "profile_only"
    DISTINCT_MODEL = "distinct_model"
    DISTINCT_PROVIDER = "distinct_provider"
    DISTINCT_PROVIDER_AND_MODEL = "distinct_provider_and_model"


@dataclass(frozen=True, slots=True)
class ProviderIndependenceResult:
    """Evaluation of provider/model independence across bindings."""

    level: ProviderIndependenceLevel
    distinct_model_identities: frozenset[str]
    distinct_provider_identities: frozenset[str]
    bindings: tuple[ProviderBindingEvidence, ...]


def _model_identity(binding: ProviderBindingEvidence) -> str:
    model = binding.model or "default"
    return f"{binding.provider}:{model}"


def _provider_identity(binding: ProviderBindingEvidence) -> str:
    return binding.provider


def evaluate_provider_independence(
    bindings: tuple[ProviderBindingEvidence, ...],
) -> ProviderIndependenceResult:
    """Compare provider, model, and profile IDs across bindings."""
    models = frozenset(_model_identity(binding) for binding in bindings)
    providers = frozenset(_provider_identity(binding) for binding in bindings)
    if len(models) >= 2 and len(providers) >= 2:
        level = ProviderIndependenceLevel.DISTINCT_PROVIDER_AND_MODEL
    elif len(models) >= 2:
        level = ProviderIndependenceLevel.DISTINCT_MODEL
    elif len(providers) >= 2:
        level = ProviderIndependenceLevel.DISTINCT_PROVIDER
    else:
        level = ProviderIndependenceLevel.PROFILE_ONLY
    return ProviderIndependenceResult(
        level=level,
        distinct_model_identities=models,
        distinct_provider_identities=providers,
        bindings=bindings,
    )


def council_requires_distinct_models(
    bindings: tuple[ProviderBindingEvidence, ...],
) -> tuple[bool, str | None]:
    """DS-E2E-02 requires at least two distinct real model identities."""
    result = evaluate_provider_independence(bindings)
    if len(result.distinct_model_identities) >= 2:
        return True, None
    models = ", ".join(sorted(result.distinct_model_identities)) or "unknown"
    return (
        False,
        "Council executed successfully but qualification requires at least two "
        f"distinct real model identities; current bindings use {models} only.",
    )


def producer_verifier_requires_distinct_models(
    producer: ProviderBindingEvidence,
    verifier: ProviderBindingEvidence,
) -> tuple[bool, str | None]:
    """DS-E2E-03 requires producer model identity != verifier model identity."""
    if _model_identity(producer) != _model_identity(verifier):
        return True, None
    model = producer.model or "default"
    return (
        False,
        "Semantic verification qualification requires producer and verifier to use "
        f"distinct model identities; both resolve to {producer.provider}/{model}.",
    )


def same_model_same_provider(
    left: ProviderBindingEvidence,
    right: ProviderBindingEvidence,
) -> bool:
    return _model_identity(left) == _model_identity(right)
