# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolution execution — apply plugin advice to uncertainty lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.evidence import (
    ExternalEffectEvidenceVerdict,
    classify_external_effect_outcome,
)
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyResolutionKind,
    UncertaintyStateRecord,
)
from intergrax.contracts.enterprise_reliability.observability import ResolutionDecisionFact
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.plugin_spi import ResolutionStrategyAdvice
from intergrax.contracts.enterprise_reliability.resolution import (
    ResolutionConsistencyError,
    ResolutionDisposition,
    assert_resolution_advice_consistent_with_evidence,
)
from intergrax.contracts.enterprise_reliability.resolution_execution import (
    ExternalEffectResolutionExecution,
    ResolutionExecutionDisposition,
    ResolutionExecutionError,
)
from intergrax.runtime.enterprise_reliability.resolution_orchestration import (
    ExternalEffectResolutionPlanning,
    ResolutionOrchestrationError,
)
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import resolve_uncertainty


class ExternalEffectResolutionRun(BaseModel):
    """Resolution bundle for recovery orchestration and observability emission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    execution: ExternalEffectResolutionExecution
    decision_fact: ResolutionDecisionFact | None = None


def _resolved_outcome_for_advice(
    advice: ResolutionStrategyAdvice,
    *,
    evidence_verdict: ExternalEffectEvidenceVerdict,
) -> ExternalEffectOutcome:
    if advice.resolution_kind is UncertaintyResolutionKind.ESCALATED:
        return ExternalEffectOutcome.UNKNOWN
    return classify_external_effect_outcome(evidence_verdict)


def execute_external_effect_resolution(
    *,
    planning: ExternalEffectResolutionPlanning,
    recorded_at: datetime | None = None,
) -> ExternalEffectResolutionRun:
    """Apply a resolution plan — closes UNKNOWN when advice is consistent with evidence."""
    plan = planning.plan
    evidence = planning.evidence
    state = planning.state
    timestamp = recorded_at or datetime.now(tz=UTC)

    if plan.disposition is ResolutionDisposition.DEFER_INSUFFICIENT_EVIDENCE:
        return ExternalEffectResolutionRun(
            state=state,
            execution=ExternalEffectResolutionExecution(
                disposition=ResolutionExecutionDisposition.DEFERRED_INSUFFICIENT_EVIDENCE,
                plan=plan,
                rationale=plan.rationale,
            ),
        )

    if plan.disposition is ResolutionDisposition.STRATEGY_UNAVAILABLE:
        return ExternalEffectResolutionRun(
            state=state,
            execution=ExternalEffectResolutionExecution(
                disposition=ResolutionExecutionDisposition.SKIPPED_PLUGIN_UNAVAILABLE,
                plan=plan,
                rationale=plan.rationale,
            ),
        )

    if plan.disposition is ResolutionDisposition.DEFER_STRATEGY_ACTION:
        return ExternalEffectResolutionRun(
            state=state,
            execution=ExternalEffectResolutionExecution(
                disposition=ResolutionExecutionDisposition.DEFERRED_STRATEGY_ACTION,
                plan=plan,
                rationale=plan.rationale,
            ),
        )

    advice = plan.advice
    if advice is None:
        raise ResolutionOrchestrationError("resolution plan missing advice")

    try:
        assert_resolution_advice_consistent_with_evidence(advice, evidence)
    except ResolutionConsistencyError as exc:
        raise ResolutionExecutionError(str(exc)) from exc

    if plan.disposition is ResolutionDisposition.ESCALATE_REQUIRED:
        disposition = ResolutionExecutionDisposition.ESCALATED_BY_POSTURE
    else:
        disposition = ResolutionExecutionDisposition.ADVICE_APPLIED

    resolved_outcome = _resolved_outcome_for_advice(
        advice,
        evidence_verdict=evidence.verdict,
    )
    try:
        updated = resolve_uncertainty(
            state,
            resolution_kind=advice.resolution_kind,
            resolved_outcome=resolved_outcome,
        )
    except Exception as exc:
        raise ResolutionExecutionError(str(exc)) from exc

    platform_action = (
        plan.decision.action
        if plan.decision is not None
        else advice.platform_action.action
        if advice.platform_action is not None
        else None
    )
    fact = ResolutionDecisionFact(
        correlation_id=state.correlation_id,
        contract_id=planning.contract_id,
        plugin_id=plan.plugin_id,
        evidence_ref=evidence.evidence_ref,
        platform_action=platform_action,
        resolution_kind=advice.resolution_kind,
        lifecycle_phase=updated.lifecycle_phase,
        recorded_at=timestamp,
    )
    return ExternalEffectResolutionRun(
        state=updated,
        execution=ExternalEffectResolutionExecution(
            disposition=disposition,
            plan=plan,
            applied_advice=advice,
            resolution_kind=advice.resolution_kind,
            rationale=advice.rationale or plan.rationale,
        ),
        decision_fact=fact,
    )


__all__ = [
    "ExternalEffectResolutionRun",
    "execute_external_effect_resolution",
]
