# © Artur Czarnecki. All rights reserved.

"""Agent promotion flow contracts and evaluator (Phase V-ALG.2).

Promotion is governance-owned. Evaluation evidence is an advisory input: it is required
as evidence, it can block (regression, threshold breach), but it can never authorize.
``PromotionDecision.governance_authorized`` is computed without any evaluation input;
``approved`` implies it (PDC-D5-R014, ADR-PDC-05).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from intergrax.contracts.evaluation_contracts import EvalComparison
from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation


class PromotionStage(str, Enum):  # noqa: UP042 - str mixin kept: str() output is part of the public contract
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
    # Typed advisory evaluation input; may block a promotion, never authorizes one.
    evaluation_comparisons: list[EvalComparison] = Field(default_factory=list)
    rollback_plan_ref: str = ""
    change_ticket_ref: str = ""


class PromotionDecision(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)
    # Governance inputs only (path, certification, rollback, change ticket).
    governance_authorized: bool = False
    # Evaluation never authorizes: always True, recorded for audit.
    evaluation_advisory_only: bool = True
    advisory_notes: list[str] = Field(default_factory=list)


def evaluate_agent_promotion(bundle: PromotionEvidenceBundle) -> PromotionDecision:
    governance_reasons: list[str] = []
    if not _is_allowed_stage_transition(bundle.source_stage, bundle.target_stage):
        governance_reasons.append(
            f"Unsupported promotion path: {bundle.source_stage.value} -> {bundle.target_stage.value}"
        )
    if not bundle.certification.eligible:
        governance_reasons.append("Agent certification is not eligible")
    if not bundle.rollback_plan_ref.strip():
        governance_reasons.append("Missing rollback plan reference")
    if not bundle.change_ticket_ref.strip():
        governance_reasons.append("Missing change ticket reference")
    governance_authorized = not governance_reasons

    evaluation_reasons: list[str] = []
    advisory_notes: list[str] = []
    refs = [ref for ref in bundle.evaluation_report_refs if ref.strip()]
    if not refs:
        evaluation_reasons.append("Missing evaluation report evidence")
    for comparison in bundle.evaluation_comparisons:
        if comparison.regressed or comparison.judge_regressed:
            evaluation_reasons.append(f"Evaluation advisory reports a regression: {comparison.candidate_run_id}")
        if comparison.threshold_breached:
            evaluation_reasons.append(f"Evaluation advisory reports a threshold breach: {comparison.candidate_run_id}")
        if comparison.clean:
            advisory_notes.append(
                f"Evaluation advisory is clean for {comparison.candidate_run_id}; "
                "advisory only, it does not authorize promotion"
            )

    reasons = governance_reasons + evaluation_reasons
    return PromotionDecision(
        approved=governance_authorized and not evaluation_reasons,
        reasons=reasons,
        governance_authorized=governance_authorized,
        advisory_notes=advisory_notes,
    )


def _is_allowed_stage_transition(source: PromotionStage, target: PromotionStage) -> bool:
    allowed_paths: set[tuple[PromotionStage, PromotionStage]] = {
        (PromotionStage.DEV, PromotionStage.STAGING),
        (PromotionStage.STAGING, PromotionStage.PRODUCTION),
    }
    return (source, target) in allowed_paths
