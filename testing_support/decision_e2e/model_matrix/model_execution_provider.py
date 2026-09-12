# © Artur Czarnecki. All rights reserved.

"""Model execution provider contracts for multi-model qualification (R6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationRuntimeIdentity,
)
from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
    OllamaProbeConfig,
    probe_ollama_runtime_identity,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.availability import ModelAvailability


class ModelProvider(Protocol):
    """Health and identity for a model endpoint."""

    def health_check(self) -> bool:
        """Return True when the provider endpoint is reachable."""

    def verify_identity(self, *, model_name: str) -> QualificationRuntimeIdentity | None:
        """Return observed runtime identity or None when unavailable."""


class ModelExecutionProvider(Protocol):
    """Execution-time model binding: probe digest and availability for a profile."""

    def resolve_profile_digest(
        self,
        profile: ModelQualificationProfile,
        *,
        env_digest: str | None,
    ) -> tuple[str | None, ModelAvailability]:
        """Resolve model digest and cohort availability for ``profile``."""


@dataclass(frozen=True, slots=True)
class OllamaModelExecutionProvider:
    """Ollama-backed ``ModelExecutionProvider``."""

    config: OllamaProbeConfig

    def health_check(self) -> bool:
        return probe_ollama_runtime_identity(self.config, model_name=None) is not None

    def verify_identity(self, *, model_name: str) -> QualificationRuntimeIdentity | None:
        return probe_ollama_runtime_identity(self.config, model_name=model_name)

    def resolve_profile_digest(
        self,
        profile: ModelQualificationProfile,
        *,
        env_digest: str | None,
    ) -> tuple[str | None, ModelAvailability]:
        if env_digest:
            return env_digest, ModelAvailability.AVAILABLE
        observed = self.verify_identity(model_name=profile.model_name)
        if observed is None or observed.model_digest is None:
            return None, ModelAvailability.MODEL_UNAVAILABLE
        return observed.model_digest, ModelAvailability.AVAILABLE


__all__ = [
    "ModelExecutionProvider",
    "ModelProvider",
    "OllamaModelExecutionProvider",
]
