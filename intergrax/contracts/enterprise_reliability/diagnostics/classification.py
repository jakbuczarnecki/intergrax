# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Public ERL reliability diagnostic severity and recommendation strategy contracts (ERL-DIAG-001D)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol, runtime_checkable

from intergrax.contracts.enterprise_reliability.diagnostics.artifact_refs import (
    ReliabilityDiagnosticArtifactRefs,
)
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    ExternalEffectReliabilityObservation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    AutomationSafetyHint,
    ExternalEffectReliabilitySignalKind,
)
from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleState,
)

ReliabilityDiagnosticSeverityStrategyId = NewType(
    "ReliabilityDiagnosticSeverityStrategyId",
    str,
)
ReliabilityDiagnosticSeverityStrategyVersion = NewType(
    "ReliabilityDiagnosticSeverityStrategyVersion",
    str,
)
ReliabilityDiagnosticRecommendationStrategyId = NewType(
    "ReliabilityDiagnosticRecommendationStrategyId",
    str,
)
ReliabilityDiagnosticRecommendationStrategyVersion = NewType(
    "ReliabilityDiagnosticRecommendationStrategyVersion",
    str,
)

RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_ID = ReliabilityDiagnosticSeverityStrategyId(
    "intergrax.diagnostics.external_effect_reliability.severity.conservative.v1",
)
RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_VERSION = ReliabilityDiagnosticSeverityStrategyVersion(
    "1",
)
RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_ID = ReliabilityDiagnosticRecommendationStrategyId(
    "intergrax.diagnostics.external_effect_reliability.recommendation.conservative.v1",
)
RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_VERSION = (
    ReliabilityDiagnosticRecommendationStrategyVersion("1")
)

MAX_RELIABILITY_CLASSIFICATION_EXPLANATION_LEN = 512


class ExternalEffectReliabilityDiagnosticSeverity(StrEnum):
    """Platform-aligned problem severity for classified reliability diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExternalEffectReliabilityOperatorRecommendationKind(StrEnum):
    """
    Advisory operator recommendation kinds for ERL diagnostics — not execution commands.

    Projection to ``DiagnosticRecommendationKind`` is owned by investigation read models (001E).
    """

    OBSERVE = "observe"
    WAIT_FOR_TRUTH = "wait_for_truth"
    REVIEW_EVIDENCE = "review_evidence"
    MANUAL_INVESTIGATION = "manual_investigation"
    ESCALATE_TO_OPERATOR = "escalate_to_operator"
    NO_OPERATOR_ACTION_REQUIRED = "no_operator_action_required"
    REQUEST_APPROVAL = "request_approval"


@dataclass(frozen=True, slots=True)
class ExternalEffectReliabilityDiagnosticClassificationContext:
    """
    Typed public input for severity and recommendation strategies.

    Built from authoritative observation facts and optional grouping subject identity.
    """

    observation: ExternalEffectReliabilityObservation
    grouping_subject_index_token: str | None = None

    @property
    def tenant_id(self) -> str:
        return self.observation.tenant_id

    @property
    def signal_kind(self) -> ExternalEffectReliabilitySignalKind:
        return self.observation.signal_kind

    @property
    def lifecycle_state(self) -> ReliabilityCaseLifecycleState:
        return self.observation.lifecycle_state

    @property
    def automation_safety_hint(self) -> AutomationSafetyHint:
        return self.observation.execution_safety_hint

    @property
    def artifact_refs(self) -> ReliabilityDiagnosticArtifactRefs:
        return self.observation.artifact_refs


@dataclass(frozen=True, slots=True)
class ExternalEffectReliabilitySeverityDecision:
    severity: ExternalEffectReliabilityDiagnosticSeverity
    strategy_id: ReliabilityDiagnosticSeverityStrategyId
    strategy_version: ReliabilityDiagnosticSeverityStrategyVersion
    reason_code: str | None = None
    safe_explanation: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalEffectReliabilityRecommendationDecision:
    recommendation_kind: ExternalEffectReliabilityOperatorRecommendationKind
    strategy_id: ReliabilityDiagnosticRecommendationStrategyId
    strategy_version: ReliabilityDiagnosticRecommendationStrategyVersion
    reason_code: str
    safe_explanation: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExternalEffectReliabilityDiagnosticClassificationResult:
    """Combined advisory classification output for downstream projection (001E)."""

    severity: ExternalEffectReliabilitySeverityDecision | None
    recommendation: ExternalEffectReliabilityRecommendationDecision | None
    severity_strategy_failed: bool = False
    recommendation_strategy_failed: bool = False


def build_reliability_diagnostic_classification_context(
    observation: ExternalEffectReliabilityObservation,
    *,
    grouping_subject_index_token: str | None = None,
) -> ExternalEffectReliabilityDiagnosticClassificationContext:
    if type(observation) is not ExternalEffectReliabilityObservation:
        raise TypeError("observation must be ExternalEffectReliabilityObservation")
    if grouping_subject_index_token is not None:
        token = grouping_subject_index_token.strip()
        if not token:
            raise ValueError("grouping_subject_index_token must be non-empty when provided")
        if grouping_subject_index_token != token:
            raise ValueError("grouping_subject_index_token must not have surrounding whitespace")
        grouping_subject_index_token = token
    return ExternalEffectReliabilityDiagnosticClassificationContext(
        observation=observation,
        grouping_subject_index_token=grouping_subject_index_token,
    )


@runtime_checkable
class ExternalEffectReliabilityDiagnosticSeverityStrategy(Protocol):
    """Pluginable severity interpretation for one reliability diagnostic context."""

    @property
    def strategy_id(self) -> ReliabilityDiagnosticSeverityStrategyId: ...

    @property
    def strategy_version(self) -> ReliabilityDiagnosticSeverityStrategyVersion: ...

    def classify(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
    ) -> ExternalEffectReliabilitySeverityDecision: ...


@runtime_checkable
class ExternalEffectReliabilityDiagnosticRecommendationStrategy(Protocol):
    """Pluginable advisory recommendation for one classified reliability context."""

    @property
    def strategy_id(self) -> ReliabilityDiagnosticRecommendationStrategyId: ...

    @property
    def strategy_version(self) -> ReliabilityDiagnosticRecommendationStrategyVersion: ...

    def recommend(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
        severity: ExternalEffectReliabilitySeverityDecision,
    ) -> ExternalEffectReliabilityRecommendationDecision: ...


__all__ = [
    "ExternalEffectReliabilityDiagnosticClassificationContext",
    "ExternalEffectReliabilityDiagnosticClassificationResult",
    "ExternalEffectReliabilityDiagnosticSeverity",
    "ExternalEffectReliabilityDiagnosticRecommendationStrategy",
    "ExternalEffectReliabilityDiagnosticSeverityStrategy",
    "ExternalEffectReliabilityOperatorRecommendationKind",
    "ExternalEffectReliabilityRecommendationDecision",
    "ExternalEffectReliabilitySeverityDecision",
    "MAX_RELIABILITY_CLASSIFICATION_EXPLANATION_LEN",
    "RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_ID",
    "RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_VERSION",
    "RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_ID",
    "RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_VERSION",
    "ReliabilityDiagnosticRecommendationStrategyId",
    "ReliabilityDiagnosticRecommendationStrategyVersion",
    "ReliabilityDiagnosticSeverityStrategyId",
    "ReliabilityDiagnosticSeverityStrategyVersion",
    "build_reliability_diagnostic_classification_context",
]
