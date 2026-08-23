# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-4: apply, rollback, canary, wiring, and governance bridge tests."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.adaptive_wiring import wire_adaptive_profile
from intergrax.applications.contracts.environment_profile import AdaptiveProfile, ApplicationEnvironmentProfile
from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
)
from intergrax.runtime.adaptive.adaptive_runtime_events import (
    build_adaptive_apply_event,
    build_adaptive_proposal_event,
    build_adaptive_rollback_event,
)
from intergrax.runtime.adaptive.canary_traffic import should_route_canary_traffic
from intergrax.runtime.adaptive.contracts import (
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionRecord,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.policy_learning_approval import (
    InMemoryPolicyLearningApprovalStore,
    PolicyLearningApprovalRequiredError,
    require_policy_learning_approval,
)
from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.adaptive.profile_mutation_store import InMemoryAdaptiveProfileMutationStore
from intergrax.runtime.adaptive.profile_pointer_store import InMemoryProfileActivePointerStore
from intergrax.runtime.adaptive.profile_policy_resolver import apply_policy_fragment_version
from intergrax.runtime.adaptive.profile_version_store import InMemoryProfileVersionStore
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopGateResult,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
)
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge
from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _policy_learning_package(*, proposal_id: str = "prop_policy") -> AdaptationProposalPackage:
    envelope = AdaptiveLoopEnvelope(
        loop_id="policy-learning-echo",
        kind=AdaptiveLoopKind.POLICY_LEARNING,
        max_iterations=3,
        max_delta_percent=15.0,
        authority=AdaptiveAuthorityLevel.AUTO_WITH_HUMAN_GATE,
        requires_human_approval=True,
        cooldown_seconds=7200,
    )
    return AdaptationProposalPackage(
        proposal_id=proposal_id,
        candidate=AdaptationProposalCandidate(
            loop_id=envelope.loop_id,
            source_engine="policy_learning",
            proposal=AdaptiveLoopProposal(
                envelope=envelope,
                proposed_change_summary="Tighten tool policy",
                human_approver_id="owner:ops",
            ),
            profile_draft=ProfileVersionDraft(
                version_id="draft-policy-echo",
                artifact_type=ProfileArtifactType.POLICY_FRAGMENT,
                artifact_payload={"deny_tool_ids": ["sandbox.exec"]},
            ),
        ),
        envelope_gate=AdaptiveLoopGateResult(loop_id=envelope.loop_id, passed=True),
        passed_all_gates=True,
    )


def _routing_package() -> AdaptationProposalPackage:
    envelope = AdaptiveLoopEnvelope(
        loop_id="routing-tuning-echo",
        kind=AdaptiveLoopKind.ROUTING_TUNING,
        max_iterations=5,
        max_delta_percent=10.0,
        authority=AdaptiveAuthorityLevel.RECOMMEND,
        requires_human_approval=False,
        cooldown_seconds=3600,
    )
    return AdaptationProposalPackage(
        candidate=AdaptationProposalCandidate(
            loop_id=envelope.loop_id,
            source_engine="routing_tuning",
            proposal=AdaptiveLoopProposal(
                envelope=envelope,
                proposed_change_summary="Route to deep tier",
            ),
            profile_draft=ProfileVersionDraft(
                version_id="draft-routing-echo",
                artifact_type=ProfileArtifactType.RAG,
                artifact_payload={"selected_arm": "rag_tier_deep"},
            ),
        ),
        envelope_gate=AdaptiveLoopGateResult(loop_id=envelope.loop_id, passed=True),
        passed_all_gates=True,
    )


def _build_executor(
    *,
    approval_store: InMemoryPolicyLearningApprovalStore | None = None,
    pointer_store: InMemoryProfileActivePointerStore | None = None,
) -> AdaptationExecutor:
    store = InMemoryProfileVersionStore()
    resolved_pointer_store = pointer_store or InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store)
    mutation_store = InMemoryAdaptiveProfileMutationStore(
        version_store=store,
        pointer_store=resolved_pointer_store,
    )
    return AdaptationExecutor(
        profile_store=store,
        pointer_store=resolved_pointer_store,
        lifecycle_manager=lifecycle,
        mutation_store=mutation_store,
        approval_store=approval_store or InMemoryPolicyLearningApprovalStore(),
    )


def test_adaptive_profile_defaults_disabled_observe() -> None:
    profile = AdaptiveProfile()
    assert profile.enabled is False
    assert profile.mode == "observe"


def test_canary_traffic_allowlist_and_percent() -> None:
    assert should_route_canary_traffic(
        tenant_id="tenant_a",
        routing_key="run-1",
        canary_tenant_allowlist=["tenant_a"],
        canary_traffic_percent=0.0,
    )
    assert should_route_canary_traffic(
        tenant_id="tenant_b",
        routing_key="run-1",
        canary_tenant_allowlist=[],
        canary_traffic_percent=100.0,
    )


def test_adaptation_executor_apply_and_rollback_roundtrip() -> None:
    executor = _build_executor()
    package_a = _routing_package()
    shadow_a = executor.shadow(package_a, tenant_id="tenant_a", task_class="echo.basic")
    first_apply = executor.apply(
        package_a,
        tenant_id="tenant_a",
        task_class="echo.basic",
        version_id=shadow_a.candidate_profile_version_id,
        expected_active_version_id=None,
    )
    assert first_apply.applied_version_id == "draft-routing-echo"

    package_b = _routing_package()
    package_b = package_b.model_copy(
        update={
            "candidate": package_b.candidate.model_copy(
                update={
                    "profile_draft": ProfileVersionDraft(
                        version_id="draft-routing-echo-v2",
                        artifact_type=ProfileArtifactType.RAG,
                        artifact_payload={"selected_arm": "rag_tier_default"},
                    )
                }
            )
        }
    )
    shadow_b = executor.shadow(package_b, tenant_id="tenant_a", task_class="echo.basic")
    second_apply = executor.apply(
        package_b,
        tenant_id="tenant_a",
        task_class="echo.basic",
        version_id=shadow_b.candidate_profile_version_id,
        expected_active_version_id="draft-routing-echo",
    )
    assert second_apply.previous_version_id == "draft-routing-echo"

    rollback = executor.rollback(
        tenant_id="tenant_a",
        task_class="echo.basic",
        artifact_type=ProfileArtifactType.RAG,
        expected_active_version_id="draft-routing-echo-v2",
    )
    assert rollback.restored_version_id == "draft-routing-echo"
    assert rollback.rolled_back_version_id == "draft-routing-echo-v2"


def test_policy_learning_apply_requires_approval() -> None:
    approval_store = InMemoryPolicyLearningApprovalStore()
    executor = _build_executor(approval_store=approval_store)
    package = _policy_learning_package()
    shadow = executor.shadow(package, tenant_id="tenant_a", task_class="echo.basic")
    with pytest.raises(PolicyLearningApprovalRequiredError):
        executor.apply(
            package,
            tenant_id="tenant_a",
            task_class="echo.basic",
            version_id=shadow.candidate_profile_version_id,
            expected_active_version_id=None,
        )
    approval_store.record_approval(package.proposal_id, approver_id="owner:ops")
    require_policy_learning_approval(package, approval_store=approval_store)
    result = executor.apply(
        package,
        tenant_id="tenant_a",
        task_class="echo.basic",
        version_id=shadow.candidate_profile_version_id,
        expected_active_version_id=None,
    )
    assert result.artifact_type == ProfileArtifactType.POLICY_FRAGMENT


def test_governance_bridge_submit_and_apply_approved() -> None:
    pointer_store = InMemoryProfileActivePointerStore()
    executor = _build_executor(pointer_store=pointer_store)
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=_AllowControlPlaneEvaluator(),
    )
    bridge = RuntimeArchitectureGovernanceBridge(
        mutation_authorization_boundary=boundary,
    )
    package = _routing_package()
    proposal_id = bridge.submit_proposal(package)
    assert proposal_id == package.proposal_id
    shadow = executor.shadow(package, tenant_id="tenant_a", task_class="echo.basic")
    applied = bridge.apply_approved(
        package,
        executor=executor,
        pointer_store=pointer_store,
        principal=RequestIdentity(
            tenant_id="tenant_a",
            user_id="operator-1",
            principal_type=PrincipalType.USER,
            auth_subject="operator-1",
        ),
        mutation_id="mut-apply-1",
        tenant_id="tenant_a",
        task_class="echo.basic",
        version_id=shadow.candidate_profile_version_id,
    )
    assert applied.apply_result.applied_version_id == "draft-routing-echo"
    assert applied.authorization_evidence.mutation_id == "mut-apply-1"


class _AllowControlPlaneEvaluator:
    def evaluate(self, request) -> PolicyDecision:
        return PolicyDecision(action=PolicyAction.ALLOW, reason="ok")


def test_apply_policy_fragment_version_attaches_domain_fragment() -> None:
    bundle = apply_policy_fragment_version(
        RuntimePolicyBundle(),
        ProfileVersionRecord(
            version_id="v-policy",
            tenant_id="tenant_a",
            artifact_type=ProfileArtifactType.POLICY_FRAGMENT,
            artifact_payload={"deny_tool_ids": ["sandbox.exec"]},
            created_by="prop",
            status=ProfileVersionStatus.ACTIVE,
        ),
    )
    assert bundle.domain_fragments["policy_fragment_version_id"] == "v-policy"


def test_adaptive_runtime_events_use_typed_event_types() -> None:
    token = bind_active_execution_identity(
        run_id="run_0123456789abcdef0123456789abcdef",
        attempt_id="attempt_0123456789abcdef0123456789abcdef",
    )
    try:
        proposal_event = build_adaptive_proposal_event(
            task_id="task_0123456789abcdef0123456789abcdef",
            run_id="run_0123456789abcdef0123456789abcdef",
            tenant_id="tenant_a",
            proposal_id="prop-1",
            loop_id="loop-1",
        )
        assert proposal_event.event_type == RuntimeEventType.DOMAIN_SIGNAL
        assert proposal_event.event_kind == "platform.adaptive.adaptive_proposal_submitted"
        apply_event = build_adaptive_apply_event(
            task_id="task_0123456789abcdef0123456789abcdef",
            run_id="run_0123456789abcdef0123456789abcdef",
            tenant_id="tenant_a",
            version_id="v1",
            artifact_type="rag",
        )
        assert apply_event.event_type == RuntimeEventType.DOMAIN_SIGNAL
        assert apply_event.event_kind == "platform.adaptive.adaptive_profile_applied"
        rollback_event = build_adaptive_rollback_event(
            task_id="task_0123456789abcdef0123456789abcdef",
            run_id="run_0123456789abcdef0123456789abcdef",
            tenant_id="tenant_a",
            restored_version_id="v0",
            artifact_type="rag",
        )
        assert rollback_event.event_type == RuntimeEventType.DOMAIN_SIGNAL
        assert rollback_event.event_kind == "platform.adaptive.adaptive_profile_rollback"
    finally:
        reset_active_execution_identity(token)


def test_wire_adaptive_profile_disabled_returns_no_executor() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    wiring = wire_adaptive_profile(env)
    assert wiring.adaptation_executor is None
    assert wiring.domain_fragments["adaptive_enabled"] is False


def test_materialize_runtime_config_includes_adaptive_profile_when_enabled(tmp_path) -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="lab.adaptive")
    env = env.model_copy(
        update={
            "adaptive_profile": AdaptiveProfile(
                enabled=True,
                mode="shadow",
                adaptive_profile_db_path=tmp_path / "adaptive_harness.db",
                signal_store_path=tmp_path / "signals.db",
            )
        }
    )
    wiring = wire_adaptive_profile(env)
    assert wiring.profile.enabled is True
    assert wiring.adaptation_executor is not None
    assert wiring.signal_collector is not None
    assert wiring.profile_version_store is not None
