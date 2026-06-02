from __future__ import annotations

from intergrax.runtime.architecture.agent_certification import (
    AgentCertificationEvidence,
    AgentCertificationGate,
    AgentCertificationOwner,
    GateCheckStatus,
    evaluate_agent_certification,
)


def test_certification_fails_without_owner_when_production_eligible() -> None:
    evidence = AgentCertificationEvidence(
        agent_id="agent.echo",
        agent_version="1.0.0",
        production_eligible=True,
        quality_gates=[
            AgentCertificationGate(
                name="unit",
                status=GateCheckStatus.PASS,
                evidence_ref="pytest -m gate",
            )
        ],
        policy_gates=[
            AgentCertificationGate(
                name="policy",
                status=GateCheckStatus.PASS,
                evidence_ref="policy evidence",
            )
        ],
        security_gates=[
            AgentCertificationGate(
                name="security",
                status=GateCheckStatus.PASS,
                evidence_ref="security evidence",
            )
        ],
    )
    result = evaluate_agent_certification(evidence)
    assert result.eligible is False
    assert any("Missing owner/on-call metadata" in reason for reason in result.reasons)


def test_certification_fails_when_any_gate_fails() -> None:
    evidence = AgentCertificationEvidence(
        agent_id="agent.echo",
        agent_version="1.0.0",
        production_eligible=True,
        owner=AgentCertificationOwner(team="platform", owner="alice", on_call="alice"),
        quality_gates=[
            AgentCertificationGate(
                name="unit",
                status=GateCheckStatus.FAIL,
                evidence_ref="pytest -m gate",
            )
        ],
        policy_gates=[
            AgentCertificationGate(
                name="policy",
                status=GateCheckStatus.PASS,
                evidence_ref="policy evidence",
            )
        ],
        security_gates=[
            AgentCertificationGate(
                name="security",
                status=GateCheckStatus.PASS,
                evidence_ref="security evidence",
            )
        ],
    )
    result = evaluate_agent_certification(evidence)
    assert result.eligible is False
    assert any("Gate failed: unit" in reason for reason in result.reasons)


def test_certification_passes_with_complete_evidence() -> None:
    evidence = AgentCertificationEvidence(
        agent_id="agent.echo",
        agent_version="1.0.0",
        production_eligible=True,
        owner=AgentCertificationOwner(team="platform", owner="alice", on_call="alice"),
        quality_gates=[
            AgentCertificationGate(
                name="unit",
                status=GateCheckStatus.PASS,
                evidence_ref="pytest -m gate",
            )
        ],
        policy_gates=[
            AgentCertificationGate(
                name="policy",
                status=GateCheckStatus.PASS,
                evidence_ref="policy evidence",
            )
        ],
        security_gates=[
            AgentCertificationGate(
                name="security",
                status=GateCheckStatus.PASS,
                evidence_ref="security evidence",
            )
        ],
    )
    result = evaluate_agent_certification(evidence)
    assert result.eligible is True
    assert result.reasons == []
