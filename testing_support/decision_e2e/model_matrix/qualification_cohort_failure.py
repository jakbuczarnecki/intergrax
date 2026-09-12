# © Artur Czarnecki. All rights reserved.

"""Failure taxonomy for R6-LIVE cohort execution (attribution, not analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from testing_support.decision_e2e.local_ai_incident_qualification import QualificationCliExit


class CohortFailureKind(StrEnum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    QUALIFICATION_FAILURE = "QUALIFICATION_FAILURE"
    ARTIFACT_FAILURE = "ARTIFACT_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


@dataclass(frozen=True, slots=True)
class CohortFailureAttribution:
    """Explicit failure ownership for pipeline consumers (CLI, analysis)."""

    kind: CohortFailureKind
    source: str
    owner: str


def failure_kind_for_exception(exc: BaseException) -> CohortFailureKind:
    """Classify isolated executor exceptions; never default unknown to provider."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return CohortFailureKind.PROVIDER_FAILURE
    if isinstance(exc, OSError) and getattr(exc, "errno", None) is not None:
        return CohortFailureKind.PROVIDER_FAILURE
    return CohortFailureKind.UNKNOWN_FAILURE


def attribution_for_exception(exc: BaseException) -> CohortFailureAttribution:
    kind = failure_kind_for_exception(exc)
    if kind is CohortFailureKind.PROVIDER_FAILURE:
        return CohortFailureAttribution(
            kind=kind,
            source="session_runner",
            owner="ModelExecutionProvider",
        )
    return CohortFailureAttribution(
        kind=kind,
        source="session_runner",
        owner="QualificationCohortExecutor",
    )


def resolve_cohort_failure(
    *,
    status: str,
    exit_code: QualificationCliExit,
) -> CohortFailureAttribution | None:
    if status == "MODEL_UNAVAILABLE":
        return CohortFailureAttribution(
            kind=CohortFailureKind.MODEL_UNAVAILABLE,
            source="resolve_profile_digest",
            owner="QualificationPlanner",
        )
    if status == "BLOCKED_PRECONDITION":
        return CohortFailureAttribution(
            kind=CohortFailureKind.QUALIFICATION_FAILURE,
            source="cohort_plan",
            owner="QualificationCohortExecutor",
        )
    if status == "ISOLATED_FAILURE":
        return CohortFailureAttribution(
            kind=CohortFailureKind.UNKNOWN_FAILURE,
            source="execute_plans",
            owner="QualificationCohortExecutor",
        )
    if status != "EXECUTED":
        return CohortFailureAttribution(
            kind=CohortFailureKind.UNKNOWN_FAILURE,
            source="cohort_execution",
            owner="QualificationCohortExecutor",
        )
    if exit_code is QualificationCliExit.SUCCESS:
        return None
    if exit_code in (
        QualificationCliExit.PARTIAL_OR_INVALID_SESSION,
        QualificationCliExit.FAILED_FINALIZATION,
    ):
        return CohortFailureAttribution(
            kind=CohortFailureKind.ARTIFACT_FAILURE,
            source="qualification_session",
            owner="LocalQualificationSession",
        )
    return CohortFailureAttribution(
        kind=CohortFailureKind.QUALIFICATION_FAILURE,
        source="qualification_session",
        owner="LocalQualificationSession",
    )


__all__ = [
    "CohortFailureAttribution",
    "CohortFailureKind",
    "attribution_for_exception",
    "failure_kind_for_exception",
    "resolve_cohort_failure",
]
