# © Artur Czarnecki. All rights reserved.

"""Model qualification profiles for multi-model alignment reliability (R6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelQualificationProfile:
    """Registry entry for one model in the qualification matrix."""

    profile_key: str
    provider: str
    model_id: str
    digest: str
    temperature: float
    expected_behavior_class: str | None = None

    def with_digest(self, digest: str) -> ModelQualificationProfile:
        return ModelQualificationProfile(
            profile_key=self.profile_key,
            provider=self.provider,
            model_id=self.model_id,
            digest=digest,
            temperature=self.temperature,
            expected_behavior_class=self.expected_behavior_class,
        )

    def artifact_dir_name(self) -> str:
        return self.profile_key


__all__ = ["ModelQualificationProfile"]
