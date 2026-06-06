# © Artur Czarnecki. All rights reserved.

"""Profile promotion evidence and evaluator (Phase W-ADAPT-3.5)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.contracts import ProfileVersionStatus
from intergrax.runtime.adaptive.profile_lifecycle import validate_profile_transition


class ProfilePromotionEvidenceBundle(BaseModel):
    """Evidence bundle for profile version promotion (mirrors agent_promotion pattern)."""

    model_config = ConfigDict(extra="forbid")

    version_id: str
    source_status: ProfileVersionStatus
    target_status: ProfileVersionStatus
    evaluation_report_refs: list[str] = Field(default_factory=list)
    rollback_plan_ref: str = ""
    change_ticket_ref: str = ""


class ProfilePromotionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_profile_promotion(
    bundle: ProfilePromotionEvidenceBundle,
) -> ProfilePromotionDecision:
    reasons: list[str] = []
    try:
        validate_profile_transition(
            current=bundle.source_status,
            target=bundle.target_status,
        )
    except ValueError as exc:
        reasons.append(str(exc))
    if not bundle.evaluation_report_refs:
        reasons.append("Missing evaluation report evidence")
    if not bundle.rollback_plan_ref:
        reasons.append("Missing rollback plan reference")
    if not bundle.change_ticket_ref:
        reasons.append("Missing change ticket reference")
    return ProfilePromotionDecision(approved=not reasons, reasons=reasons)
