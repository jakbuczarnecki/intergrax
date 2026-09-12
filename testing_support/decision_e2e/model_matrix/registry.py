# © Artur Czarnecki. All rights reserved.

"""Configurable model matrix registry (no hardcoded models in proof scripts)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile

_MATRIX_VERSION = "r6-v1"


def qualification_matrix_version() -> str:
    return _MATRIX_VERSION


def iter_qualification_profiles() -> tuple[ModelQualificationProfile, ...]:
    """First R6 iteration matrix; extend here without changing proof pipeline."""
    return (
        ModelQualificationProfile(
            profile_key="qwen2.5-14b",
            provider="ollama",
            model_id="qwen2.5:14b",
            digest="",
            temperature=0.0,
            expected_behavior_class="baseline",
        ),
        ModelQualificationProfile(
            profile_key="qwen2.5-32b",
            provider="ollama",
            model_id="qwen2.5:32b",
            digest="",
            temperature=0.0,
            expected_behavior_class="reasoning",
        ),
        ModelQualificationProfile(
            profile_key="llama",
            provider="ollama",
            model_id="llama3.1:8b",
            digest="",
            temperature=0.0,
            expected_behavior_class="alternative_alignment",
        ),
    )


def profile_by_key(profile_key: str) -> ModelQualificationProfile | None:
    for profile in iter_qualification_profiles():
        if profile.profile_key == profile_key:
            return profile
    return None


__all__ = [
    "iter_qualification_profiles",
    "profile_by_key",
    "qualification_matrix_version",
]
