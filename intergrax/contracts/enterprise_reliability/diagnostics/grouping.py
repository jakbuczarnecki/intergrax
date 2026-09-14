# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Public ERL reliability Problem grouping contracts (ERL-DIAG-001C)."""

from __future__ import annotations

from typing import NewType, Protocol, runtime_checkable

from intergrax.contracts.diagnostics.subject_ref import ProblemGroupingSubjectRef
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    ExternalEffectReliabilityObservation,
)

ReliabilityProblemGroupingStrategyId = NewType(
    "ReliabilityProblemGroupingStrategyId",
    str,
)
ReliabilityProblemGroupingStrategyVersion = NewType(
    "ReliabilityProblemGroupingStrategyVersion",
    str,
)

RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID = ReliabilityProblemGroupingStrategyId(
    "intergrax.diagnostics.external_effect_reliability.case.v1",
)
RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION = ReliabilityProblemGroupingStrategyVersion(
    "1",
)

_RELIABILITY_CASE_INDEX_PREFIX = "erl:case:"
_OCCURRENCE_SEGMENT = ":obs:"


@runtime_checkable
class ReliabilityCaseSubjectRef(ProblemGroupingSubjectRef, Protocol):
    """Grouping subject for one tenant-scoped ERL reliability case."""

    @property
    def reliability_case_id(self) -> str: ...


@runtime_checkable
class ExternalEffectReliabilityProblemGroupingStrategy(Protocol):
    """
    Pluginable grouping: maps one observation to a stable Problem grouping subject.

    Does not allocate ``ProblemId`` and must not participate in occurrence identity.
    """

    @property
    def strategy_id(self) -> ReliabilityProblemGroupingStrategyId: ...

    @property
    def strategy_version(self) -> ReliabilityProblemGroupingStrategyVersion: ...

    def group(
        self,
        observation: ExternalEffectReliabilityObservation,
    ) -> ReliabilityCaseSubjectRef: ...


def reliability_case_subject_index_token(reliability_case_id: str) -> str:
    """Deterministic grouping index token — tenant is applied at persistence boundary."""
    normalized = _require_semantic_identifier(
        reliability_case_id,
        field_name="reliability_case_id",
    )
    return f"{_RELIABILITY_CASE_INDEX_PREFIX}{normalized}"


def reliability_diagnostic_occurrence_instance_id(
    *,
    reliability_case_id: str,
    observation_id: str,
) -> str:
    """
    Orchestration scope ``instance_id`` — encodes case (grouping) + observation (occurrence).

    Distinct ``observation_id`` values yield distinct occurrence subjects under one Problem.
    """
    case_id = _require_semantic_identifier(
        reliability_case_id,
        field_name="reliability_case_id",
    )
    obs_id = _require_semantic_identifier(observation_id, field_name="observation_id")
    if _OCCURRENCE_SEGMENT in case_id:
        raise ValueError("reliability_case_id must not contain occurrence segment")
    return f"{_RELIABILITY_CASE_INDEX_PREFIX}{case_id}{_OCCURRENCE_SEGMENT}{obs_id}"


def parse_reliability_diagnostic_occurrence_instance_id(
    instance_id: str,
) -> tuple[str, str]:
    """Return ``(reliability_case_id, observation_id)`` from a scope instance id."""
    normalized = _require_semantic_identifier(instance_id, field_name="instance_id")
    if not normalized.startswith(_RELIABILITY_CASE_INDEX_PREFIX):
        raise ValueError("instance_id is not an ERL reliability diagnostic subject")
    remainder = normalized[len(_RELIABILITY_CASE_INDEX_PREFIX) :]
    if _OCCURRENCE_SEGMENT not in remainder:
        raise ValueError("instance_id missing observation segment")
    case_id, obs_id = remainder.split(_OCCURRENCE_SEGMENT, 1)
    if not case_id or not obs_id:
        raise ValueError("instance_id must encode non-empty case and observation")
    if _OCCURRENCE_SEGMENT in case_id:
        raise ValueError("reliability_case_id must not contain occurrence segment")
    return case_id, obs_id


def _require_semantic_identifier(value: str, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty and not whitespace-only")
    if value != normalized:
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")
    return normalized


__all__ = [
    "ExternalEffectReliabilityProblemGroupingStrategy",
    "RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID",
    "RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION",
    "ReliabilityCaseSubjectRef",
    "ReliabilityProblemGroupingStrategyId",
    "ReliabilityProblemGroupingStrategyVersion",
    "parse_reliability_diagnostic_occurrence_instance_id",
    "reliability_case_subject_index_token",
    "reliability_diagnostic_occurrence_instance_id",
]
