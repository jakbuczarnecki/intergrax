# © Artur Czarnecki. All rights reserved.

"""Agent promotion flow contracts and evaluator (Phase V-ALG.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation


class PromotionStage(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class PromotionEvidenceBundle(BaseModel):
    agent_id: str
    agent_version: str
    source_stage: PromotionStage
    target_stage: PromotionStage
    certification: AgentCertificationEvaluation
    evaluation_report_refs: list[str] = Field(default_factory=list)
    rollback_plan_ref: str = ""
    change_ticket_ref: str = ""


class PromotionDecision(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_agent_promotion(bundle: PromotionEvidenceBundle) -> PromotionDecision:
    reasons: list[str] = []
    if not _is_allowed_stage_transition(bundle.source_stage, bundle.target_stage):
        reasons.append(
            f"Unsupported promotion path: {bundle.source_stage.value} -> {bundle.target_stage.value}"
        )
    if not bundle.certification.eligible:
        reasons.append("Agent certification is not eligible")
    if not bundle.evaluation_report_refs:
        reasons.append("Missing evaluation report evidence")
    if not bundle.rollback_plan_ref:
        reasons.append("Missing rollback plan reference")
    if not bundle.change_ticket_ref:
        reasons.append("Missing change ticket reference")
    return PromotionDecision(approved=not reasons, reasons=reasons)


def _is_allowed_stage_transition(source: PromotionStage, target: PromotionStage) -> bool:
    allowed_paths: set[tuple[PromotionStage, PromotionStage]] = {
        (PromotionStage.DEV, PromotionStage.STAGING),
        (PromotionStage.STAGING, PromotionStage.PRODUCTION),
    }
    return (source, target) in allowed_paths
