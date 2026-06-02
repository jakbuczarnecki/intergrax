from __future__ import annotations

from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
    build_default_adaptive_proposals,
    evaluate_adaptive_governance,
    evaluate_bounded_adaptive_loop,
)


def test_default_adaptive_proposals_pass_bounded_governance() -> None:
    report = evaluate_adaptive_governance(build_default_adaptive_proposals())
    assert report.passed is True


def test_policy_learning_requires_human_approver() -> None:
    result = evaluate_bounded_adaptive_loop(
        AdaptiveLoopProposal(
            envelope=AdaptiveLoopEnvelope(
                loop_id="policy-bad",
                kind=AdaptiveLoopKind.POLICY_LEARNING,
                max_iterations=2,
                max_delta_percent=10.0,
                authority=AdaptiveAuthorityLevel.AUTO_WITH_HUMAN_GATE,
                requires_human_approval=True,
            ),
            proposed_change_summary="Unsafe auto policy change",
        )
    )
    assert result.passed is False
