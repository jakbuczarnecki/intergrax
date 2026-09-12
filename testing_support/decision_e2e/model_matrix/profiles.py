# © Artur Czarnecki. All rights reserved.

"""Model qualification profiles for multi-model alignment reliability (R6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelQualificationProfile:
    """Immutable registry entry for one model in the qualification matrix."""

    profile_id: str
    profile_key: str
    provider: str
    model_name: str
    digest: str
    runtime_version: str
    temperature: float
    evaluator_iterations: int
    revision_budget: int
    expected_behavior_class: str | None = None

    def with_digest(self, digest: str) -> ModelQualificationProfile:
        return ModelQualificationProfile(
            profile_id=self.profile_id,
            profile_key=self.profile_key,
            provider=self.provider,
            model_name=self.model_name,
            digest=digest,
            runtime_version=self.runtime_version,
            temperature=self.temperature,
            evaluator_iterations=self.evaluator_iterations,
            revision_budget=self.revision_budget,
            expected_behavior_class=self.expected_behavior_class,
        )

    def artifact_dir_name(self) -> str:
        return self.profile_key


__all__ = ["ModelQualificationProfile"]
