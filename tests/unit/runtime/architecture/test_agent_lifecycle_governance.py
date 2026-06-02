from __future__ import annotations

from intergrax.runtime.architecture.agent_lifecycle_governance import (
    AgentLifecycleState,
    AgentLifecycleTransitionRequest,
    compute_deprecation_deadline,
    evaluate_agent_lifecycle_transition,
)


def test_lifecycle_rejects_retirement_without_deprecated_state() -> None:
    decision = evaluate_agent_lifecycle_transition(
        AgentLifecycleTransitionRequest(
            agent_id="agent:research",
            agent_version="1.0.0",
            current_state=AgentLifecycleState.PRODUCTION,
            target_state=AgentLifecycleState.RETIRED,
            migration_window_days=30,
            migration_guide_ref="runbook/migrations/research.md",
            deprecation_notice_ref="notices/research.md",
        )
    )
    assert decision.approved is False
    assert any("Retirement requires deprecated state" in reason for reason in decision.reasons)


def test_lifecycle_accepts_deprecation_with_complete_evidence() -> None:
    decision = evaluate_agent_lifecycle_transition(
        AgentLifecycleTransitionRequest(
            agent_id="agent:research",
            agent_version="1.0.0",
            current_state=AgentLifecycleState.PRODUCTION,
            target_state=AgentLifecycleState.DEPRECATED,
            migration_window_days=30,
            migration_guide_ref="runbook/migrations/research.md",
            deprecation_notice_ref="notices/research.md",
        )
    )
    assert decision.approved is True
    assert decision.reasons == []


def test_compute_deprecation_deadline_is_after_request_time() -> None:
    request = AgentLifecycleTransitionRequest(
        agent_id="agent:research",
        agent_version="1.0.0",
        current_state=AgentLifecycleState.PRODUCTION,
        target_state=AgentLifecycleState.DEPRECATED,
        migration_window_days=21,
        migration_guide_ref="runbook/migrations/research.md",
        deprecation_notice_ref="notices/research.md",
    )
    deadline = compute_deprecation_deadline(request)
    assert deadline > request.requested_at
