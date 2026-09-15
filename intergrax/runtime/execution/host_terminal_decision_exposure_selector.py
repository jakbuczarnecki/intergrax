# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default host-terminal Decision exposure selection (scope/policy deterministic)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_authoritative_exposure import DecisionEvaluationScope
from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionDecision,
    DecisionExposureSelectionFailure,
    DecisionExposureSelectionFailureCode,
    DecisionExposureSelectionStrategy,
    DecisionExposureSelectionSuccessReason,
    HostPublicationClass,
)
from intergrax.runtime.execution.decision_exposure_selection_validation import (
    validate_single_attempt_partition_for_selection,
)

HOST_TERMINAL_DECISION_EXPOSURE_SELECTOR_ID = "platform.host_terminal_decision_exposure"


@dataclass(frozen=True, slots=True)
class _TerminalSubjectKey:
    evaluation_scope: DecisionEvaluationScope
    namespace: str
    subject: str


def _candidate_sort_key(
    candidate: DecisionExposureCandidate[object],
) -> tuple[int, str, str, str]:
    return (
        candidate.evaluation_ordinal,
        candidate.evaluation_scope.value,
        candidate.decision_scope.namespace,
        candidate.decision_scope.subject,
    )


class HostTerminalDecisionExposureSelector:
    """Reusable default strategy: eligible host-terminal candidates only."""

    @property
    def strategy_id(self) -> str:
        return HOST_TERMINAL_DECISION_EXPOSURE_SELECTOR_ID

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object] | DecisionExposureSelectionFailure:
        if type(policy) is not DecisionExposurePublicationPolicy:
            raise TypeError("policy must be DecisionExposurePublicationPolicy")
        attempt_check = validate_single_attempt_partition_for_selection(candidates)
        if type(attempt_check) is DecisionExposureSelectionFailure:
            return attempt_check
        eligible = [
            candidate
            for candidate in candidates
            if candidate.host_publication_class is HostPublicationClass.HOST_TERMINAL_CANDIDATE
            and candidate.evaluation_scope in policy.eligible_terminal_scopes
        ]
        if not eligible:
            return DecisionExposureSelectionFailure(
                reason_code=DecisionExposureSelectionFailureCode.NO_ELIGIBLE_TERMINAL_CANDIDATE,
                detail="no host-terminal candidate matches publication policy",
            )
        eligible_sorted = sorted(eligible, key=_candidate_sort_key)
        by_subject: dict[_TerminalSubjectKey, list[DecisionExposureCandidate[object]]] = {}
        for candidate in eligible_sorted:
            key = _TerminalSubjectKey(
                evaluation_scope=candidate.evaluation_scope,
                namespace=candidate.decision_scope.namespace,
                subject=candidate.decision_scope.subject,
            )
            by_subject.setdefault(key, []).append(candidate)
        for group in by_subject.values():
            if len(group) > 1:
                return DecisionExposureSelectionFailure(
                    reason_code=DecisionExposureSelectionFailureCode.AMBIGUOUS_TERMINAL_CANDIDATES,
                    detail="multiple equivalent terminal candidates for one subject",
                )
        if len(by_subject) != 1:
            return DecisionExposureSelectionFailure(
                reason_code=DecisionExposureSelectionFailureCode.AMBIGUOUS_TERMINAL_CANDIDATES,
                detail="multiple terminal subjects without host disambiguation policy",
            )
        selected = next(iter(by_subject.values()))[0]
        return DecisionExposureSelectionDecision(
            selected=selected.exposure,
            reason_code=DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE,
            considered_candidates=len(candidates),
        )


def default_decision_exposure_selection_strategy() -> DecisionExposureSelectionStrategy[object]:
    """Built-in platform selector (works without external plugins)."""
    return HostTerminalDecisionExposureSelector()
