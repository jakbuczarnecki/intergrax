# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolution orchestration contracts (ERL — foundation)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.enterprise_reliability.effect_contract import (
    UnknownUncertaintyPosture,
)
from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyResolutionKind
from intergrax.contracts.enterprise_reliability.plugin_spi import ResolutionStrategyAdvice
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
    ExternalEffectEvidenceConfidence,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    ResolutionDecision,
    ResolutionPlatformAction,
    abstained_resolution_decision,
    missing_resolution_strategy_decision,
)

SCHEMA_RESOLUTION_PLAN_V1: Final = "resolution_plan.v1"


class ResolutionDisposition(StrEnum):
    """Platform decision before applying lifecycle closure — no recovery I/O here."""

    INVOKE_PLUGIN = "invoke_plugin"
    DEFER_INSUFFICIENT_EVIDENCE = "defer_insufficient_evidence"
    ESCALATE_REQUIRED = "escalate_required"
    STRATEGY_UNAVAILABLE = "strategy_unavailable"
    DEFER_STRATEGY_ACTION = "defer_strategy_action"


class ResolutionPlanningError(ValueError):
    """Resolution plan cannot be derived from evidence and posture."""


class ResolutionConsistencyError(ValueError):
    """Plugin advice conflicts with recorded reconciliation evidence."""


class ResolutionPlan(BaseModel):
    """Non-executing resolution intent — plugin selection and advice snapshot only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RESOLUTION_PLAN_V1
    disposition: ResolutionDisposition
    decision: ResolutionDecision | None = None
    plugin_id: str | None = None
    advice: ResolutionStrategyAdvice | None = None
    rationale: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _validate_plugin_fields(self) -> ResolutionPlan:
        if self.disposition is ResolutionDisposition.INVOKE_PLUGIN:
            if self.decision is None:
                raise ValueError("decision required when disposition is invoke_plugin")
            if self.plugin_id is None or not self.plugin_id.strip():
                raise ValueError("plugin_id required when disposition is invoke_plugin")
            if self.advice is None:
                raise ValueError("advice required when disposition is invoke_plugin")
        elif self.disposition is ResolutionDisposition.ESCALATE_REQUIRED:
            if self.plugin_id is not None:
                raise ValueError("plugin_id forbidden when disposition is escalate_required")
            if self.decision is None:
                raise ValueError("decision required when disposition is escalate_required")
        elif self.disposition is ResolutionDisposition.STRATEGY_UNAVAILABLE:
            if self.decision is None:
                raise ValueError("decision required when disposition is strategy_unavailable")
            if self.plugin_id is not None or self.advice is not None:
                raise ValueError(
                    "plugin_id and advice forbidden when resolution strategy is unavailable",
                )
        elif self.disposition is ResolutionDisposition.DEFER_STRATEGY_ACTION:
            if self.decision is None:
                raise ValueError("decision required when disposition defers strategy action")
            if self.advice is not None:
                raise ValueError("advice forbidden when disposition defers strategy action")
        else:
            if (
                self.plugin_id is not None
                or self.advice is not None
                or self.decision is not None
            ):
                raise ValueError(
                    "plugin_id, advice, and decision forbidden when disposition defers resolution",
                )
        return self


def evaluate_resolution_disposition(
    *,
    unknown_posture: UnknownUncertaintyPosture,
    evidence: ExternalEffectEvidence,
) -> ResolutionDisposition:
    """
    Classify whether resolution may consult a plugin strategy.

    Does not invoke plugins or mutate lifecycle.
    """
    if unknown_posture is UnknownUncertaintyPosture.ESCALATE_REQUIRED:
        return ResolutionDisposition.ESCALATE_REQUIRED
    if evidence.confidence is not ExternalEffectEvidenceConfidence.DEFINITIVE:
        return ResolutionDisposition.DEFER_INSUFFICIENT_EVIDENCE
    if evidence.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT:
        return ResolutionDisposition.DEFER_INSUFFICIENT_EVIDENCE
    return ResolutionDisposition.INVOKE_PLUGIN


def resolution_advice_from_decision(
    decision: ResolutionDecision,
    evidence: ExternalEffectEvidence,
) -> ResolutionStrategyAdvice | None:
    """Map a typed platform decision to lifecycle closure advice when applicable."""
    action = decision.action
    if action is ResolutionPlatformAction.ESCALATE:
        return ResolutionStrategyAdvice(
            resolution_kind=UncertaintyResolutionKind.ESCALATED,
            rationale=decision.rationale,
            platform_action=decision,
        )
    if action is ResolutionPlatformAction.STOP:
        return ResolutionStrategyAdvice(
            resolution_kind=UncertaintyResolutionKind.CONFIRMED_FAILURE,
            rationale=decision.rationale,
            platform_action=decision,
        )
    if action is ResolutionPlatformAction.CONTINUE:
        if evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
            return ResolutionStrategyAdvice(
                resolution_kind=UncertaintyResolutionKind.CONFIRMED_SUCCESS,
                rationale=decision.rationale,
                platform_action=decision,
            )
        if evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
            return ResolutionStrategyAdvice(
                resolution_kind=UncertaintyResolutionKind.CONFIRMED_FAILURE,
                rationale=decision.rationale,
                platform_action=decision,
            )
        return None
    if action in (
        ResolutionPlatformAction.COMPENSATION_REQUIRED,
        ResolutionPlatformAction.UNKNOWN,
    ):
        return None
    raise ResolutionPlanningError(f"unsupported resolution platform action: {action.value}")


def assert_resolution_advice_consistent_with_evidence(
    advice: ResolutionStrategyAdvice,
    evidence: ExternalEffectEvidence,
) -> None:
    """Fail closed when plugin advice contradicts definitive reconciliation evidence."""
    if evidence.confidence is not ExternalEffectEvidenceConfidence.DEFINITIVE:
        raise ResolutionConsistencyError(
            "resolution advice requires definitive reconciliation evidence",
        )
    if advice.resolution_kind is UncertaintyResolutionKind.ESCALATED:
        return
    if advice.resolution_kind is UncertaintyResolutionKind.CONFIRMED_SUCCESS:
        if evidence.verdict is not ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
            raise ResolutionConsistencyError(
                "confirmed_success resolution requires definitive_success evidence",
            )
        return
    if advice.resolution_kind is UncertaintyResolutionKind.CONFIRMED_FAILURE:
        if evidence.verdict is not ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
            raise ResolutionConsistencyError(
                "confirmed_failure resolution requires definitive_failure evidence",
            )
        return
    raise ResolutionConsistencyError(f"unsupported resolution_kind: {advice.resolution_kind.value}")


def build_resolution_plan(
    *,
    unknown_posture: UnknownUncertaintyPosture,
    evidence: ExternalEffectEvidence,
    plugin_id: str,
    decision: ResolutionDecision,
    strategy_registered: bool,
    rationale: str = "",
) -> ResolutionPlan:
    """Materialize a resolution plan after strategy selection and evidence validation."""
    posture_disposition = evaluate_resolution_disposition(
        unknown_posture=unknown_posture,
        evidence=evidence,
    )
    if posture_disposition is ResolutionDisposition.ESCALATE_REQUIRED:
        posture_decision = ResolutionDecision(
            action=ResolutionPlatformAction.ESCALATE,
            rationale=rationale or posture_disposition.value,
        )
        return ResolutionPlan(
            disposition=posture_disposition,
            decision=posture_decision,
            advice=ResolutionStrategyAdvice(
                resolution_kind=UncertaintyResolutionKind.ESCALATED,
                rationale=posture_decision.rationale,
                platform_action=posture_decision,
            ),
            rationale=posture_decision.rationale,
        )
    if posture_disposition is ResolutionDisposition.DEFER_INSUFFICIENT_EVIDENCE:
        return ResolutionPlan(
            disposition=posture_disposition,
            rationale=rationale or posture_disposition.value,
        )
    normalized_plugin = plugin_id.strip()
    if not normalized_plugin:
        raise ResolutionPlanningError("plugin_id required")
    if not strategy_registered:
        missing = missing_resolution_strategy_decision()
        return ResolutionPlan(
            disposition=ResolutionDisposition.STRATEGY_UNAVAILABLE,
            decision=missing,
            rationale=missing.rationale,
        )
    if decision.action is ResolutionPlatformAction.CONTINUE and not rationale:
        rationale = decision.rationale
    advice = resolution_advice_from_decision(decision, evidence)
    if advice is None:
        deferred = (
            decision
            if decision.action
            in (
                ResolutionPlatformAction.COMPENSATION_REQUIRED,
                ResolutionPlatformAction.UNKNOWN,
            )
            else abstained_resolution_decision()
        )
        return ResolutionPlan(
            disposition=ResolutionDisposition.DEFER_STRATEGY_ACTION,
            decision=deferred,
            plugin_id=normalized_plugin,
            rationale=deferred.rationale,
        )
    assert_resolution_advice_consistent_with_evidence(advice, evidence)
    return ResolutionPlan(
        disposition=ResolutionDisposition.INVOKE_PLUGIN,
        decision=decision,
        plugin_id=normalized_plugin,
        advice=advice,
        rationale=rationale or decision.rationale or advice.rationale,
    )


__all__ = [
    "ResolutionConsistencyError",
    "ResolutionDisposition",
    "ResolutionPlan",
    "ResolutionPlanningError",
    "SCHEMA_RESOLUTION_PLAN_V1",
    "assert_resolution_advice_consistent_with_evidence",
    "build_resolution_plan",
    "ResolutionDecision",
    "ResolutionPlatformAction",
    "evaluate_resolution_disposition",
    "resolution_advice_from_decision",
]
