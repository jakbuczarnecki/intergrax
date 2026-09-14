# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default ERL reliability-case Problem grouping strategy (ERL-DIAG-001C)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.diagnostics.grouping import (
    RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID,
    RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION,
    ExternalEffectReliabilityProblemGroupingStrategy,
    ReliabilityCaseSubjectRef,
    ReliabilityProblemGroupingStrategyId,
    ReliabilityProblemGroupingStrategyVersion,
    parse_reliability_diagnostic_occurrence_instance_id,
    reliability_case_subject_index_token,
)
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    ExternalEffectReliabilityObservation,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingCandidate,
    ProblemGroupingInput,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyResult,
    ProblemGroupingStrategyVersion,
    ProblemGroupingSubject,
    ProblemGroupingSubjectFindingSource,
    ProblemGroupingSubjectRef,
)
from intergrax.runtime.diagnostics.reliability.observation_to_problem_signal import (
    ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID,
)
from intergrax.runtime.diagnostics.reliability.reliability_case_grouping_reconciliation import (
    ReliabilityCaseProblemGroupingBasis,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY,
)

STRATEGY_ID = ProblemGroupingStrategyId(str(RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID))
STRATEGY_VERSION = ProblemGroupingStrategyVersion(
    str(RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION),
)


@dataclass(frozen=True, slots=True)
class _DefaultReliabilityCaseSubjectRef:
    tenant_id: str
    reliability_case_id: str

    @property
    def index_token(self) -> str:
        return reliability_case_subject_index_token(self.reliability_case_id)


class ReliabilityCaseDefaultObservationGroupingStrategy:
    """Public SPI default — tenant + reliability case → stable grouping subject."""

    @property
    def strategy_id(self) -> ReliabilityProblemGroupingStrategyId:
        return RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID

    @property
    def strategy_version(self) -> ReliabilityProblemGroupingStrategyVersion:
        return RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION

    def group(
        self,
        observation: ExternalEffectReliabilityObservation,
    ) -> ReliabilityCaseSubjectRef:
        if type(observation) is not ExternalEffectReliabilityObservation:
            raise TypeError("observation must be ExternalEffectReliabilityObservation")
        return _DefaultReliabilityCaseSubjectRef(
            tenant_id=observation.tenant_id,
            reliability_case_id=observation.reliability_case_id,
        )


class ReliabilityCaseDefaultGroupingStrategy:
    """
    Batch grouping over normalized ERL signal assessments.

    Groups by ``reliability_case_id`` while preserving per-observation subject refs
    for occurrence identity (encoded in orchestration ``instance_id``).
    """

    def __init__(
        self,
        *,
        observation_grouping: ExternalEffectReliabilityProblemGroupingStrategy | None = None,
    ) -> None:
        self._observation_grouping = (
            observation_grouping or ReliabilityCaseDefaultObservationGroupingStrategy()
        )

    @property
    def strategy_id(self) -> ProblemGroupingStrategyId:
        return STRATEGY_ID

    @property
    def strategy_version(self) -> ProblemGroupingStrategyVersion:
        return STRATEGY_VERSION

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics:
        return ProblemGroupingStrategyCharacteristics(
            method=ProblemGroupingMethod.DETERMINISTIC,
            deterministic=True,
            requires_features=False,
        )

    def group(
        self,
        inputs: tuple[ProblemGroupingInput, ...],
    ) -> ProblemGroupingStrategyResult:
        buckets: dict[str, list[ProblemGroupingSubjectRef]] = {}
        case_order: list[str] = []

        for input_item in inputs:
            parsed = _parse_erl_reliability_grouping_subject(input_item.subject)
            if parsed is None:
                continue
            case_id, subject_ref = parsed
            members = buckets.get(case_id)
            if members is None:
                members = []
                buckets[case_id] = members
                case_order.append(case_id)
            members.append(subject_ref)

        candidates: list[ProblemGroupingCandidate] = []
        for case_id in case_order:
            members = tuple(buckets[case_id])
            candidates.append(
                ProblemGroupingCandidate(
                    members=members,
                    provenance=ProblemGroupingProvenance(
                        strategy_id=self.strategy_id,
                        strategy_version=self.strategy_version,
                        method=ProblemGroupingMethod.DETERMINISTIC,
                        supporting_subject_refs=members,
                        basis=ReliabilityCaseProblemGroupingBasis(
                            reliability_case_id=case_id,
                        ),
                    ),
                ),
            )

        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=tuple(candidates),
        )


def _parse_erl_reliability_grouping_subject(
    subject: ProblemGroupingSubject,
) -> tuple[str, ProblemGroupingSubjectRef] | None:
    application = subject.ref.application_instance()
    if application is None:
        return None
    if application.application_id != ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID:
        return None
    if not subject.findings:
        return None
    for finding in subject.findings:
        if finding.source is not ProblemGroupingSubjectFindingSource.PLATFORM_SIGNAL:
            return None
        if finding.problem_kind != PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY:
            return None
    try:
        case_id, _observation_id = parse_reliability_diagnostic_occurrence_instance_id(
            application.instance_id,
        )
    except ValueError:
        return None
    return case_id, subject.ref


__all__ = [
    "ReliabilityCaseDefaultGroupingStrategy",
    "ReliabilityCaseDefaultObservationGroupingStrategy",
    "STRATEGY_ID",
    "STRATEGY_VERSION",
]
