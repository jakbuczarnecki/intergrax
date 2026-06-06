# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-3: profile version store, lifecycle, executor, and router tests."""

from __future__ import annotations

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
)
from intergrax.runtime.adaptive.contracts import (
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionRecord,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.profile_lifecycle import (
    ProfileLifecycleTransitionError,
    ProfileVersionLifecycleManager,
    validate_profile_transition,
)
from intergrax.runtime.adaptive.profile_promotion import (
    ProfilePromotionEvidenceBundle,
    evaluate_profile_promotion,
)
from intergrax.runtime.adaptive.profile_rag_router import ProfileAwareQueryRouter
from intergrax.runtime.adaptive.profile_pointer_store import InMemoryProfileActivePointerStore
from intergrax.runtime.adaptive.profile_version_store import (
    InMemoryProfileVersionStore,
    SQLiteProfileVersionStore,
)
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopGateResult,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
)
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sample_draft() -> ProfileVersionDraft:
    return ProfileVersionDraft(
        version_id="draft-routing-echo-rag_tier_deep",
        artifact_type=ProfileArtifactType.RAG,
        artifact_payload={"selected_arm": "rag_tier_deep"},
        created_by="routing_tuning",
    )


def _sample_package(*, passed: bool = True) -> AdaptationProposalPackage:
    envelope = AdaptiveLoopEnvelope(
        loop_id="routing-tuning-echo.basic",
        kind=AdaptiveLoopKind.ROUTING_TUNING,
        max_iterations=5,
        max_delta_percent=10.0,
        authority=AdaptiveAuthorityLevel.RECOMMEND,
        requires_human_approval=False,
        cooldown_seconds=3600,
    )
    candidate = AdaptationProposalCandidate(
        loop_id=envelope.loop_id,
        source_engine="routing_tuning",
        proposal=AdaptiveLoopProposal(
            envelope=envelope,
            proposed_change_summary="Recommend routing shift toward rag_tier_deep",
        ),
        profile_draft=_sample_draft(),
    )
    return AdaptationProposalPackage(
        candidate=candidate,
        envelope_gate=AdaptiveLoopGateResult(
            loop_id=envelope.loop_id,
            passed=True,
            reasons=[],
        ),
        passed_all_gates=passed,
    )


def test_profile_version_store_create_and_list() -> None:
    store = InMemoryProfileVersionStore()
    draft = _sample_draft()
    record = store.create_from_draft(draft, tenant_id="tenant_a", task_class="echo.basic")
    assert record.status == ProfileVersionStatus.DRAFT
    assert record.tenant_id == "tenant_a"
    listed = store.list_versions(tenant_id="tenant_a", status=ProfileVersionStatus.DRAFT)
    assert len(listed) == 1


def test_profile_version_store_rejects_duplicate_version_id() -> None:
    store = InMemoryProfileVersionStore()
    draft = _sample_draft()
    store.create_from_draft(draft, tenant_id="tenant_a")
    with pytest.raises(ValueError, match="already exists"):
        store.create_from_draft(draft, tenant_id="tenant_a")


def test_profile_version_store_payload_is_immutable_on_status_update() -> None:
    store = InMemoryProfileVersionStore()
    record = store.create_from_draft(_sample_draft(), tenant_id="tenant_a")
    mutated = record.model_copy(
        update={"artifact_payload": {"selected_arm": "changed"}, "status": ProfileVersionStatus.SHADOW}
    )
    with pytest.raises(ValueError, match="immutable"):
        store.save_status(mutated)


def test_profile_lifecycle_valid_transitions() -> None:
    validate_profile_transition(
        current=ProfileVersionStatus.DRAFT,
        target=ProfileVersionStatus.SHADOW,
    )
    validate_profile_transition(
        current=ProfileVersionStatus.SHADOW,
        target=ProfileVersionStatus.CANARY,
    )


def test_profile_lifecycle_rejects_invalid_transition() -> None:
    with pytest.raises(ProfileLifecycleTransitionError):
        validate_profile_transition(
            current=ProfileVersionStatus.DRAFT,
            target=ProfileVersionStatus.ACTIVE,
        )


def test_profile_lifecycle_manager_promotes_draft_to_shadow() -> None:
    store = InMemoryProfileVersionStore()
    lifecycle = ProfileVersionLifecycleManager(store)
    record = store.create_from_draft(_sample_draft(), tenant_id="tenant_a", task_class="echo.basic")
    updated = lifecycle.transition(record.version_id, target=ProfileVersionStatus.SHADOW)
    assert updated.status == ProfileVersionStatus.SHADOW


def test_adaptation_executor_shadow_allocates_candidate_version() -> None:
    store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store)
    executor = AdaptationExecutor(
        profile_store=store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
    )
    package = _sample_package()
    result = executor.shadow(package, tenant_id="tenant_a", task_class="echo.basic")
    assert result.candidate_profile_version_id == "draft-routing-echo-rag_tier_deep"
    assert result.trace_tag == "candidate_profile_version_id"
    stored = store.get(result.candidate_profile_version_id)
    assert stored is not None
    assert stored.status == ProfileVersionStatus.SHADOW


def test_adaptation_executor_shadow_rejects_failed_gate() -> None:
    store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store)
    executor = AdaptationExecutor(
        profile_store=store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
    )
    with pytest.raises(ValueError, match="failed governance"):
        executor.shadow(_sample_package(passed=False), tenant_id="tenant_a", task_class="echo.basic")


def test_profile_promotion_requires_evidence() -> None:
    decision = evaluate_profile_promotion(
        ProfilePromotionEvidenceBundle(
            version_id="v1",
            source_status=ProfileVersionStatus.SHADOW,
            target_status=ProfileVersionStatus.CANARY,
        )
    )
    assert decision.approved is False
    assert "Missing evaluation report evidence" in decision.reasons


def test_profile_promotion_approves_with_full_evidence() -> None:
    decision = evaluate_profile_promotion(
        ProfilePromotionEvidenceBundle(
            version_id="v1",
            source_status=ProfileVersionStatus.SHADOW,
            target_status=ProfileVersionStatus.CANARY,
            evaluation_report_refs=["eval/report.json"],
            rollback_plan_ref="runbooks/rollback-routing.md",
            change_ticket_ref="AHIA-123",
        )
    )
    assert decision.approved is True


def test_profile_aware_query_router_uses_candidate_arm() -> None:
    base = RagProfile(deep_query_min_words=12)
    candidate = ProfileVersionRecord(
        version_id="v-shadow",
        tenant_id="tenant_a",
        task_class="echo.basic",
        artifact_type=ProfileArtifactType.RAG,
        artifact_payload={"selected_arm": "rag_tier_deep"},
        created_by="prop_1",
        status=ProfileVersionStatus.SHADOW,
    )
    router = ProfileAwareQueryRouter(base, candidate_version=candidate)
    assert router.route("one two three four five six seven eight nine ten eleven") == "deep"


def test_governance_bridge_records_candidate_profile_version_on_shadow_observation() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    bridge = RuntimeArchitectureGovernanceBridge(evaluation_registry=registry)
    observation = bridge.record_shadow_run_evaluation(
        run_id="run-shadow-1",
        agent_id="echo",
        scenario_id="harness.smoke",
        passed=True,
        score=0.88,
        candidate_profile_version_id="draft-routing-echo-rag_tier_deep",
    )
    assert observation.candidate_profile_version_id == "draft-routing-echo-rag_tier_deep"
    metadata = bridge.build_trace_metadata(candidate_profile_version_id="draft-routing-echo-rag_tier_deep")
    assert metadata.candidate_profile_version_id == "draft-routing-echo-rag_tier_deep"


def test_sqlite_profile_version_store_roundtrip(tmp_path) -> None:
    store = SQLiteProfileVersionStore(db_path=tmp_path / "profiles.db")
    record = store.create_from_draft(_sample_draft(), tenant_id="tenant_sql", task_class="echo.basic")
    restored = store.get(record.version_id)
    assert restored is not None
    assert restored.tenant_id == "tenant_sql"
