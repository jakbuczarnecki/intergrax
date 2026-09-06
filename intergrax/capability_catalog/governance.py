# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable governance narrowing over Stage-4 ranked candidates (Stage 5)."""

from __future__ import annotations

from typing import Final, Protocol

from intergrax.capability_catalog.errors import CapabilityGovernanceError
from intergrax.capability_catalog.governed_candidate import (
    BlockedCapabilityCandidate,
    GovernedCapabilityCandidate,
)
from intergrax.capability_catalog.governed_result import GovernedDiscoveryResult
from intergrax.capability_catalog.governance_validation import validate_governed_output
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.governance import (
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
    NORMATIVE_CAPABILITY_GOVERNANCE_REASON_CODES,
)

AVAILABILITY_PRESERVING_GOVERNANCE_EVALUATOR_ID: Final = "baseline.availability"


class CapabilityGovernanceDecision:
    """Immutable governance decision from one evaluator."""

    __slots__ = ("disposition", "evidence")

    def __init__(
        self,
        *,
        disposition: GovernanceDisposition,
        evidence: GovernanceDecisionEvidence,
    ) -> None:
        self.disposition = disposition
        self.evidence = evidence


class CapabilityGovernanceEvaluator(Protocol):
    """Structural governance plugin — narrowing only, never selection."""

    @property
    def evaluator_id(self) -> str:
        """Stable evaluator identifier."""

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        """Return a governance disposition for one ranked candidate."""


class AvailabilityPreservingGovernanceEvaluator:
    """Baseline evaluator — preserves Stage-3 availability, never elevates."""

    @property
    def evaluator_id(self) -> str:
        return AVAILABILITY_PRESERVING_GOVERNANCE_EVALUATOR_ID

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        availability = candidate.availability
        if availability is AvailabilityDisposition.BLOCKED:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.AVAILABILITY_BLOCKED,
                ),
            )
        if availability is AvailabilityDisposition.UNAVAILABLE:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.AVAILABILITY_UNAVAILABLE,
                ),
            )
        if availability is AvailabilityDisposition.SCOPE_UNAVAILABLE:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=(
                        CapabilityGovernanceReasonCode.AVAILABILITY_SCOPE_UNAVAILABLE
                    ),
                ),
            )
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


def _validate_evaluator_decision(
    *,
    decision: CapabilityGovernanceDecision,
    evaluator_id: str,
) -> None:
    if decision.evidence.evaluator_id != evaluator_id:
        raise CapabilityGovernanceError(
            "governance evaluator output evidence.evaluator_id must match evaluator_id",
        )
    if decision.evidence.disposition != decision.disposition:
        raise CapabilityGovernanceError(
            "governance evaluator output evidence.disposition must match decision",
        )
    if decision.evidence.reason_code not in NORMATIVE_CAPABILITY_GOVERNANCE_REASON_CODES:
        raise CapabilityGovernanceError(
            "governance evaluator output reason_code is not a normative reason code",
        )


def _evaluator_failure_decision(
    evaluator_id: str,
    *,
    detail: str,
) -> CapabilityGovernanceDecision:
    return CapabilityGovernanceDecision(
        disposition=GovernanceDisposition.BLOCKED,
        evidence=GovernanceDecisionEvidence(
            evaluator_id=evaluator_id,
            disposition=GovernanceDisposition.BLOCKED,
            reason_code=CapabilityGovernanceReasonCode.EVALUATOR_FAILURE,
            detail=detail,
        ),
    )


def _validate_governance_pipeline_configuration(
    evaluators: tuple[CapabilityGovernanceEvaluator, ...],
    context: CapabilityGovernanceContext,
) -> None:
    if (
        context.posture is CapabilityGovernancePosture.STRICT
        and not evaluators
    ):
        raise CapabilityGovernanceError(
            "STRICT capability governance requires at least one evaluator",
        )

    seen_evaluator_ids: set[str] = set()
    for evaluator in evaluators:
        try:
            evaluator_id = require_non_empty_text(
                evaluator.evaluator_id,
                label="evaluator_id",
            )
        except (TypeError, ValueError) as exc:
            raise CapabilityGovernanceError(str(exc)) from exc
        if evaluator_id in seen_evaluator_ids:
            raise CapabilityGovernanceError(
                "governance pipeline evaluator_id values must be unique",
            )
        seen_evaluator_ids.add(evaluator_id)


def _evaluate_candidate(
    candidate: RankedCapabilityCandidate,
    evaluators: tuple[CapabilityGovernanceEvaluator, ...],
    context: CapabilityGovernanceContext,
) -> tuple[GovernanceDisposition, tuple[GovernanceDecisionEvidence, ...]]:
    evidence_items: list[GovernanceDecisionEvidence] = []
    blocked = False

    for evaluator in evaluators:
        evaluator_id = evaluator.evaluator_id
        try:
            decision = evaluator.evaluate(candidate, context)
            _validate_evaluator_decision(
                decision=decision,
                evaluator_id=evaluator_id,
            )
        except CapabilityGovernanceError:
            raise
        except Exception as exc:  # noqa: BLE001 - evaluator failure must fail closed
            if context.posture is CapabilityGovernancePosture.STRICT:
                failure = _evaluator_failure_decision(
                    evaluator_id,
                    detail=str(exc),
                )
                evidence_items.append(failure.evidence)
                blocked = True
                continue
            raise CapabilityGovernanceError(
                f"governance evaluator {evaluator_id!r} failed: {exc}",
            ) from exc

        evidence_items.append(decision.evidence)
        if decision.disposition is GovernanceDisposition.BLOCKED:
            blocked = True

    if blocked:
        return GovernanceDisposition.BLOCKED, tuple(evidence_items)
    return GovernanceDisposition.ALLOWED, tuple(evidence_items)


def govern_capability_candidates(
    ranked_candidates: tuple[RankedCapabilityCandidate, ...],
    *,
    evaluators: tuple[CapabilityGovernanceEvaluator, ...],
    context: CapabilityGovernanceContext | None = None,
) -> GovernedDiscoveryResult:
    """Run governance evaluators and partition ranked candidates fail-closed."""
    governance_context = context or CapabilityGovernanceContext()
    _validate_governance_pipeline_configuration(evaluators, governance_context)
    allowed: list[GovernedCapabilityCandidate] = []
    blocked: list[BlockedCapabilityCandidate] = []

    for candidate in ranked_candidates:
        disposition, evidence = _evaluate_candidate(
            candidate,
            evaluators,
            governance_context,
        )
        if disposition is GovernanceDisposition.BLOCKED:
            blocked.append(
                BlockedCapabilityCandidate(
                    ranked=candidate,
                    evidence=evidence,
                ),
            )
        else:
            allowed.append(
                GovernedCapabilityCandidate(
                    ranked=candidate,
                    evidence=evidence,
                ),
            )

    result = GovernedDiscoveryResult(
        allowed=tuple(allowed),
        blocked=tuple(blocked),
    )
    validate_governed_output(
        input_ranked=ranked_candidates,
        result=result,
    )
    return result
