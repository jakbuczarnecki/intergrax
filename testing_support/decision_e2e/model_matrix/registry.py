# © Artur Czarnecki. All rights reserved.

"""Configurable model matrix registry (no hardcoded models in proof scripts)."""

from __future__ import annotations

from testing_support.decision_e2e.local_ai_incident_qualification import (
    R4R1_EVALUATOR_MAX_ITERATIONS,
    R4R1_MAX_DECISION_REVISIONS,
    R4R1_RUNTIME_VERSION,
    R4R1_TEMPERATURE,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile

_MATRIX_VERSION = "r6-v2"
_RUNTIME_VERSION = R4R1_RUNTIME_VERSION.normalized()


def qualification_matrix_version() -> str:
    return _MATRIX_VERSION


def _default_profile(
    *,
    profile_key: str,
    model_name: str,
    expected_behavior_class: str | None,
) -> ModelQualificationProfile:
    return ModelQualificationProfile(
        profile_id=profile_key,
        profile_key=profile_key,
        provider="ollama",
        model_name=model_name,
        digest="",
        runtime_version=_RUNTIME_VERSION,
        temperature=R4R1_TEMPERATURE,
        evaluator_iterations=R4R1_EVALUATOR_MAX_ITERATIONS,
        revision_budget=R4R1_MAX_DECISION_REVISIONS,
        expected_behavior_class=expected_behavior_class,
    )


def iter_qualification_profiles() -> tuple[ModelQualificationProfile, ...]:
    """Matrix entries; extend here without changing runner, analyzer, or CLI pipeline."""
    return (
        _default_profile(
            profile_key="qwen2.5-14b",
            model_name="qwen2.5:14b",
            expected_behavior_class="baseline",
        ),
        _default_profile(
            profile_key="qwen2.5-32b",
            model_name="qwen2.5:32b",
            expected_behavior_class="reasoning",
        ),
        _default_profile(
            profile_key="llama3.1-8b",
            model_name="llama3.1:8b",
            expected_behavior_class="alternative_alignment",
        ),
    )


class QualificationRegistry:
    """Lookup facade over the frozen profile matrix."""

    @staticmethod
    def version() -> str:
        return qualification_matrix_version()

    @staticmethod
    def profiles() -> tuple[ModelQualificationProfile, ...]:
        return iter_qualification_profiles()

    @staticmethod
    def resolve(profile_key: str) -> ModelQualificationProfile | None:
        return profile_by_key(profile_key)


def profile_by_key(profile_key: str) -> ModelQualificationProfile | None:
    for profile in iter_qualification_profiles():
        if profile.profile_key == profile_key:
            return profile
    return None


__all__ = [
    "QualificationRegistry",
    "iter_qualification_profiles",
    "profile_by_key",
    "qualification_matrix_version",
]
