# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-3.6: shadow run records observation with candidate profile version."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
)
from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionDraft
from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
from intergrax.runtime.adaptive.profile_pointer_store import InMemoryProfileActivePointerStore
from intergrax.runtime.adaptive.profile_version_store import InMemoryProfileVersionStore
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopGateResult,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
)
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def test_shadow_allocation_records_candidate_profile_version_observation() -> None:
    store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store)
    executor = AdaptationExecutor(
        profile_store=store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
    )

    envelope = AdaptiveLoopEnvelope(
        loop_id="routing-tuning-echo.basic",
        kind=AdaptiveLoopKind.ROUTING_TUNING,
        max_iterations=5,
        max_delta_percent=10.0,
        authority=AdaptiveAuthorityLevel.RECOMMEND,
        requires_human_approval=False,
        cooldown_seconds=3600,
    )
    package = AdaptationProposalPackage(
        candidate=AdaptationProposalCandidate(
            loop_id=envelope.loop_id,
            source_engine="routing_tuning",
            proposal=AdaptiveLoopProposal(
                envelope=envelope,
                proposed_change_summary="Shadow routing candidate",
            ),
            profile_draft=ProfileVersionDraft(
                version_id="draft-shadow-integration",
                artifact_type=ProfileArtifactType.RAG,
                artifact_payload={"selected_arm": "rag_tier_deep"},
                created_by="integration_test",
            ),
        ),
        envelope_gate=AdaptiveLoopGateResult(loop_id=envelope.loop_id, passed=True),
        passed_all_gates=True,
    )

    allocation = executor.shadow(package, tenant_id="tenant_shadow", task_class="echo.basic")
    registry = InMemoryOnlineEvaluationRegistry()
    bridge = RuntimeArchitectureGovernanceBridge(evaluation_registry=registry)
    observation = bridge.record_shadow_run_evaluation(
        run_id="run-shadow-integration",
        agent_id="echo",
        scenario_id="harness.smoke",
        passed=True,
        score=0.91,
        candidate_profile_version_id=allocation.candidate_profile_version_id,
    )

    assert observation.candidate_profile_version_id == "draft-shadow-integration"
    assert registry.list_observations()[0].candidate_profile_version_id == "draft-shadow-integration"
    stored = store.get(allocation.candidate_profile_version_id)
    assert stored is not None
    assert stored.status.value == "shadow"
