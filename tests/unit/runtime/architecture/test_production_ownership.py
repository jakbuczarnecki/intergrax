from __future__ import annotations

from intergrax.runtime.architecture.production_ownership import (
    ProductionOwnerMetadata,
    ProductionOwnershipEvidence,
    evaluate_production_ownership,
)


def test_production_ownership_rejects_missing_owner() -> None:
    decision = evaluate_production_ownership(
        ProductionOwnershipEvidence(
            agent_id="agent:research",
            agent_version="1.0.0",
            production_eligible=True,
            owner=None,
            runbook_ref="runbook/agents/research.md",
        )
    )
    assert decision.approved is False
    assert any("Missing owner metadata" in reason for reason in decision.reasons)


def test_production_ownership_accepts_complete_evidence() -> None:
    decision = evaluate_production_ownership(
        ProductionOwnershipEvidence(
            agent_id="agent:research",
            agent_version="1.0.0",
            production_eligible=True,
            owner=ProductionOwnerMetadata(
                team="harness-platform",
                owner="alice",
                on_call="alice-oncall",
                escalation_channel="#harness-oncall",
            ),
            runbook_ref="runbook/agents/research.md",
        )
    )
    assert decision.approved is True
    assert decision.reasons == []
