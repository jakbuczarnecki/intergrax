#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-AHI.2 — bounded policy learning without governance drift."""

from __future__ import annotations

import sys

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
    ProfileVersionDraft,
)
from intergrax.runtime.adaptive.contracts import ProfileArtifactType
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopProposal
from intergrax.runtime.adaptive.bounded_policy_learning import evaluate_bounded_policy_learning
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopGateResult,
    AdaptiveLoopKind,
)


def _policy_learning_package(*, with_approver: bool = True) -> AdaptationProposalPackage:
    envelope = AdaptiveLoopEnvelope(
        loop_id="policy-learning-test",
        kind=AdaptiveLoopKind.POLICY_LEARNING,
        max_iterations=3,
        max_delta_percent=10.0,
        authority=AdaptiveAuthorityLevel.AUTO_WITH_HUMAN_GATE,
        requires_human_approval=True,
        cooldown_seconds=3600,
    )
    return AdaptationProposalPackage(
        proposal_id="prop_test",
        candidate=AdaptationProposalCandidate(
            loop_id=envelope.loop_id,
            source_engine="policy_learning",
            proposal=AdaptiveLoopProposal(
                envelope=envelope,
                proposed_change_summary="Tighten policy fragment",
                human_approver_id="owner:ops" if with_approver else None,
            ),
            profile_draft=ProfileVersionDraft(
                version_id="draft-1",
                artifact_type=ProfileArtifactType.POLICY_FRAGMENT,
                artifact_payload={"deny_tool_ids": []},
            ),
        ),
        envelope_gate=AdaptiveLoopGateResult(loop_id=envelope.loop_id, passed=True),
        passed_all_gates=True,
    )


def main() -> int:
    ok_report = evaluate_bounded_policy_learning(_policy_learning_package(with_approver=True))
    if not ok_report.bounded:
        print(f"valid policy learning package must be bounded: {ok_report.reasons}", file=sys.stderr)
        return 1

    bad_report = evaluate_bounded_policy_learning(_policy_learning_package(with_approver=False))
    if bad_report.bounded:
        print("policy learning without approver must not be bounded", file=sys.stderr)
        return 1

    print("OK: bounded policy learning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
