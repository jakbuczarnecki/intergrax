# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator-facing Decision context read DTOs (DIAG R4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationKind,
)
from intergrax.contracts.decision_identity import DecisionId
from intergrax.contracts.execution_identity import AttemptId, ExecutionId


class DecisionContextReadStatus(StrEnum):
    """Whether related decision context could be attached for one occurrence."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class DecisionContextUnavailableReason(StrEnum):
    """Expected absence of decision enrichment — execution diagnostics still valid."""

    NO_CORRELATION_EVIDENCE = "no_correlation_evidence"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NON_EXECUTION_SUBJECT = "non_execution_subject"


@dataclass(frozen=True, slots=True)
class DecisionContextFact:
    """Bounded contextual fact from Decision System — not diagnostic certainty."""

    key: str
    value: str


@dataclass(frozen=True, slots=True)
class RelatedDecisionContextEntry:
    """One related decision identity plus optional contextual facts."""

    decision_id: DecisionId
    decision_version: int
    decision_attempt_id: AttemptId
    execution_id: ExecutionId | None
    correlation_kind: DecisionExecutionCorrelationKind
    contextual_facts: tuple[DecisionContextFact, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionContextView:
    """
    Related decision context for operator read.

    Never assigns root cause; decisions are contextual evidence only.
    """

    read_status: DecisionContextReadStatus
    related_decisions: tuple[RelatedDecisionContextEntry, ...]
    unavailable_reason: DecisionContextUnavailableReason | None = None
    limitations: tuple[str, ...] = ()

    @staticmethod
    def unavailable(
        reason: DecisionContextUnavailableReason,
        *,
        limitations: tuple[str, ...] = (),
    ) -> DecisionContextView:
        return DecisionContextView(
            read_status=DecisionContextReadStatus.UNAVAILABLE,
            related_decisions=(),
            unavailable_reason=reason,
            limitations=limitations,
        )

    @staticmethod
    def degraded(
        related_decisions: tuple[RelatedDecisionContextEntry, ...],
        *,
        limitations: tuple[str, ...],
    ) -> DecisionContextView:
        return DecisionContextView(
            read_status=DecisionContextReadStatus.DEGRADED,
            related_decisions=related_decisions,
            unavailable_reason=DecisionContextUnavailableReason.PROVIDER_UNAVAILABLE,
            limitations=limitations,
        )
