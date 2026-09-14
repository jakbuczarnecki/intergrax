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
from intergrax.runtime.diagnostics.reliability.reliability_observation_grouping_adapter import (
    ReliabilityObservationGroupingAdapterError,
    grouping_subject_index_token_for_erl_signal_scope,
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

    Buckets by the public observation grouping SPI ``index_token`` (when provided
    upstream) while preserving per-observation subject refs for occurrence identity.
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
        bucket_order: list[str] = []

        for input_item in inputs:
            resolved = _resolve_erl_grouping_bucket(input_item, self._observation_grouping)
            if resolved is None:
                continue
            bucket_key, subject_ref = resolved
            members = buckets.get(bucket_key)
            if members is None:
                members = []
                buckets[bucket_key] = members
                bucket_order.append(bucket_key)
            members.append(subject_ref)

        candidates: list[ProblemGroupingCandidate] = []
        for bucket_key in bucket_order:
            members = tuple(buckets[bucket_key])
            candidates.append(
                ProblemGroupingCandidate(
                    members=members,
                    provenance=ProblemGroupingProvenance(
                        strategy_id=self.strategy_id,
                        strategy_version=self.strategy_version,
                        method=ProblemGroupingMethod.DETERMINISTIC,
                        supporting_subject_refs=members,
                        basis=ReliabilityCaseProblemGroupingBasis(
                            grouping_subject_index_token=bucket_key,
                        ),
                    ),
                ),
            )

        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=tuple(candidates),
        )


def _resolve_erl_grouping_bucket(
    input_item: ProblemGroupingInput,
    observation_grouping: ExternalEffectReliabilityProblemGroupingStrategy,
) -> tuple[str, ProblemGroupingSubjectRef] | None:
    subject = input_item.subject
    subject_ref = _parse_erl_reliability_subject_ref(subject)
    if subject_ref is None:
        return None

    token = subject.grouping_subject_index_token
    if token is None:
        token = _resolve_grouping_token_from_signals(input_item, observation_grouping)
    if token is None:
        return None

    application = subject.ref.application_instance()
    if application is None:
        return None
    if application.tenant_id != subject.tenant_id:
        raise ValueError("ERL grouping subject tenant mismatch")

    return token, subject_ref


def _resolve_grouping_token_from_signals(
    input_item: ProblemGroupingInput,
    observation_grouping: ExternalEffectReliabilityProblemGroupingStrategy,
) -> str | None:
    signals = input_item.signal_source_signals
    if not signals:
        return None
    subject = input_item.subject
    application = subject.ref.application_instance()
    if application is None:
        return None
    try:
        return grouping_subject_index_token_for_erl_signal_scope(
            tenant_id=application.tenant_id,
            application_id=application.application_id,
            instance_id=application.instance_id,
            problem_signals=signals,
            strategy=observation_grouping,
        )
    except ReliabilityObservationGroupingAdapterError:
        return None


def _parse_erl_reliability_subject_ref(
    subject: ProblemGroupingSubject,
) -> ProblemGroupingSubjectRef | None:
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
    return subject.ref


__all__ = [
    "ReliabilityCaseDefaultGroupingStrategy",
    "ReliabilityCaseDefaultObservationGroupingStrategy",
    "STRATEGY_ID",
    "STRATEGY_VERSION",
]
