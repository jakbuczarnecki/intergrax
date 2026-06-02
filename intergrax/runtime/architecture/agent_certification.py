# © Artur Czarnecki. All rights reserved.

"""Agent certification evidence contracts and gate evaluator (Phase V-ALG.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class GateCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class AgentCertificationOwner(BaseModel):
    team: str
    owner: str
    on_call: str


class AgentCertificationGate(BaseModel):
    name: str
    status: GateCheckStatus
    evidence_ref: str
    details: str = ""


class AgentCertificationEvidence(BaseModel):
    agent_id: str
    agent_version: str
    production_eligible: bool = False
    owner: AgentCertificationOwner | None = None
    quality_gates: list[AgentCertificationGate] = Field(default_factory=list)
    policy_gates: list[AgentCertificationGate] = Field(default_factory=list)
    security_gates: list[AgentCertificationGate] = Field(default_factory=list)


class AgentCertificationEvaluation(BaseModel):
    agent_id: str
    agent_version: str
    eligible: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_agent_certification(
    evidence: AgentCertificationEvidence,
) -> AgentCertificationEvaluation:
    reasons: list[str] = []
    if evidence.production_eligible and evidence.owner is None:
        reasons.append("Missing owner/on-call metadata for production-eligible agent")

    if not evidence.quality_gates:
        reasons.append("Missing quality gates")
    if not evidence.policy_gates:
        reasons.append("Missing policy gates")
    if not evidence.security_gates:
        reasons.append("Missing security gates")

    for gate in [*evidence.quality_gates, *evidence.policy_gates, *evidence.security_gates]:
        if gate.status != GateCheckStatus.PASS:
            reasons.append(f"Gate failed: {gate.name}")

    return AgentCertificationEvaluation(
        agent_id=evidence.agent_id,
        agent_version=evidence.agent_version,
        eligible=len(reasons) == 0,
        reasons=reasons,
    )
