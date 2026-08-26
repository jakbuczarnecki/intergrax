# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Deterministic structural problem grouping strategy (DIAG-5B).

Groups executions that share exactly the same typed diagnostic structure.
Conservative, high-precision baseline — no fuzzy similarity, ML, or LLM.
"""

from __future__ import annotations

from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicFindingSignature,
    DeterministicLimitationSignature,
    DeterministicProblemGroupingBasis,
    DeterministicProblemSignature,
    ProblemGroupingCandidate,
    ProblemGroupingInput,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyResult,
    ProblemGroupingStrategyVersion,
    ProblemGroupingSubject,
    ProblemGroupingSubjectFinding,
    ProblemGroupingSubjectLimitation,
    ProblemGroupingSubjectRef,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleViolationTransition

STRATEGY_ID = ProblemGroupingStrategyId("intergrax.diagnostics.structural.v1")
STRATEGY_VERSION = ProblemGroupingStrategyVersion("1")

_TRANSITION_ABSENT = (0,)
_TRANSITION_PRESENT_PREFIX = (1,)


def _lifecycle_transition_sort_key(
    transition: LifecycleViolationTransition | None,
) -> tuple[object, ...]:
    if transition is None:
        return _TRANSITION_ABSENT
    return (
        *_TRANSITION_PRESENT_PREFIX,
        transition.violation_kind.value,
        transition.prior_status.value,
        transition.violating_event_type.value,
    )


def _finding_sort_key(finding: ProblemGroupingSubjectFinding) -> tuple[object, ...]:
    return (
        finding.kind.value,
        finding.scope.value,
        finding.source_anomaly_kind.value,
        _lifecycle_transition_sort_key(finding.lifecycle_transition),
    )


def _limitation_sort_key(
    limitation: ProblemGroupingSubjectLimitation,
) -> tuple[object, ...]:
    return (
        limitation.kind.value,
        limitation.source_anomaly_kind.value,
    )


def _finding_signature(finding: ProblemGroupingSubjectFinding) -> DeterministicFindingSignature:
    return DeterministicFindingSignature(
        kind=finding.kind,
        scope=finding.scope,
        source_anomaly_kind=finding.source_anomaly_kind,
        lifecycle_transition=finding.lifecycle_transition,
    )


def _limitation_signature(
    limitation: ProblemGroupingSubjectLimitation,
) -> DeterministicLimitationSignature:
    return DeterministicLimitationSignature(
        kind=limitation.kind,
        source_anomaly_kind=limitation.source_anomaly_kind,
    )


def build_deterministic_problem_signature(
    subject: ProblemGroupingSubject,
) -> DeterministicProblemSignature:
    """Derive the canonical structural signature for one grouping subject."""
    sorted_findings = tuple(
        _finding_signature(finding)
        for finding in sorted(subject.findings, key=_finding_sort_key)
    )
    sorted_limitations = tuple(
        _limitation_signature(limitation)
        for limitation in sorted(subject.limitations, key=_limitation_sort_key)
    )
    return DeterministicProblemSignature(
        findings=sorted_findings,
        limitations=sorted_limitations,
    )


class DeterministicProblemGroupingStrategy:
    """
    Production deterministic structural grouping strategy (DIAG-5B).

    Changing signature field semantics requires a strategy_version bump so
    stable problem lifecycle (DIAG-5D) can reason about grouping evolution.
    """

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
        buckets: dict[DeterministicProblemSignature, list[ProblemGroupingSubjectRef]] = {}
        signature_first_seen: list[DeterministicProblemSignature] = []

        for input_item in inputs:
            subject = input_item.subject
            if not subject.findings:
                continue

            signature = build_deterministic_problem_signature(subject)
            members = buckets.get(signature)
            if members is None:
                members = []
                buckets[signature] = members
                signature_first_seen.append(signature)
            members.append(subject.ref)

        candidates: list[ProblemGroupingCandidate] = []
        for signature in signature_first_seen:
            members = tuple(buckets[signature])
            candidates.append(
                ProblemGroupingCandidate(
                    members=members,
                    provenance=ProblemGroupingProvenance(
                        strategy_id=self.strategy_id,
                        strategy_version=self.strategy_version,
                        method=ProblemGroupingMethod.DETERMINISTIC,
                        supporting_subject_refs=members,
                        basis=DeterministicProblemGroupingBasis(signature=signature),
                    ),
                )
            )

        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=tuple(candidates),
        )
