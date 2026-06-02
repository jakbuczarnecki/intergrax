# © Artur Czarnecki. All rights reserved.

"""Production ownership guard contracts (Phase V-ALG.4)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProductionOwnerMetadata(BaseModel):
    team: str
    owner: str
    on_call: str
    escalation_channel: str


class ProductionOwnershipEvidence(BaseModel):
    agent_id: str
    agent_version: str
    production_eligible: bool
    owner: ProductionOwnerMetadata | None = None
    runbook_ref: str = ""


class ProductionOwnershipDecision(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_production_ownership(evidence: ProductionOwnershipEvidence) -> ProductionOwnershipDecision:
    reasons: list[str] = []
    if not evidence.production_eligible:
        return ProductionOwnershipDecision(approved=True, reasons=[])

    if evidence.owner is None:
        reasons.append("Missing owner metadata for production-eligible agent")
    if not evidence.runbook_ref:
        reasons.append("Missing runbook reference for production-eligible agent")
    return ProductionOwnershipDecision(approved=not reasons, reasons=reasons)
