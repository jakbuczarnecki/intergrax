# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Classifier input observation contract (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.decision_system.qualification.signals import (
    EnvironmentQualificationSignal,
    EvaluatorQualificationSignal,
    ModelBehaviorQualificationSignal,
    ObservabilityQualificationSignal,
    PlatformContractQualificationSignal,
    ProviderQualificationSignal,
)
from intergrax.decision_system.qualification.taxonomy import DecisionFailureBoundary


@dataclass(frozen=True, slots=True)
class DecisionQualificationObservation:
    """Immutable structured facts for deterministic failure classification."""

    boundary: DecisionFailureBoundary
    platform_contract: PlatformContractQualificationSignal
    model_behavior: ModelBehaviorQualificationSignal
    evaluator: EvaluatorQualificationSignal
    provider: ProviderQualificationSignal
    environment: EnvironmentQualificationSignal
    observability: ObservabilityQualificationSignal
    observability_complete: bool = True
