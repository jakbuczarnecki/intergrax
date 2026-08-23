# © Artur Czarnecki. All rights reserved.

"""AHI control-plane mutation governance proofs (AHICPM1–AHICPM25)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    control_plane_mutation_request_digest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
)
from intergrax.runtime.adaptive.control_plane_governance import (
    AhiGovernanceBlockedError,
    build_apply_profile_mutation_request,
    build_rollback_profile_mutation_request,
    MUTATION_TYPE_APPLY_PROFILE,
    MUTATION_TYPE_ROLLBACK_PROFILE,
)
from intergrax.runtime.adaptive.contracts import (
    HarnessOutcomeSignal,
    OutcomeEvalMode,
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.policy_learning_approval import InMemoryPolicyLearningApprovalStore
from intergrax.runtime.adaptive.profile_lifecycle import (
    ProfileLifecycleTransitionError,
    ProfileVersionLifecycleManager,
)
from intergrax.runtime.adaptive.profile_mutation_store import (
    InMemoryAdaptiveProfileMutationStore,
    SQLiteAdaptiveProfileMutationStore,
)
from intergrax.runtime.adaptive.profile_pointer_store import (
    InMemoryProfileActivePointerStore,
    ProfileActivePointerConflictError,
    SQLiteProfileActivePointerStore,
)
from intergrax.runtime.adaptive.profile_version_store import (
    InMemoryProfileVersionStore,
    SQLiteProfileVersionStore,
)
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.adaptive.verification_loop import VerificationLoop
from intergrax.runtime.adaptive.verification_models import VerificationContext, VerificationTarget
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveLoopEnvelope,
    AdaptiveLoopGateResult,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
)
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_TASK_CLASS = "echo.basic"
_VERSION_ID = "draft-routing-echo"
_PROPOSAL_ID = "prop_routing_01"


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)
    raise_error: bool = False

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        if self.raise_error:
            raise RuntimeError("evaluator exploded")
        return self.decision


def _routing_package() -> AdaptationProposalPackage:
    envelope = AdaptiveLoopEnvelope(
        loop_id="routing-tuning-echo.basic",
        kind=AdaptiveLoopKind.ROUTING_TUNING,
        max_iterations=5,
        max_delta_percent=10.0,
        authority=AdaptiveAuthorityLevel.RECOMMEND,
        requires_human_approval=False,
        cooldown_seconds=3600,
    )
    return AdaptationProposalPackage(
        proposal_id=_PROPOSAL_ID,
        candidate=AdaptationProposalCandidate(
            loop_id=envelope.loop_id,
            source_engine="routing_tuning",
            proposal=AdaptiveLoopProposal(
                envelope=envelope,
                proposed_change_summary="Recommend routing shift",
            ),
            profile_draft=ProfileVersionDraft(
                version_id=_VERSION_ID,
                artifact_type=ProfileArtifactType.RAG,
                artifact_payload={"selected_arm": "rag_tier_deep"},
            ),
        ),
        envelope_gate=AdaptiveLoopGateResult(loop_id=envelope.loop_id, passed=True),
        passed_all_gates=True,
    )


def _user_principal(tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="operator-1",
        principal_type=PrincipalType.USER,
        auth_subject="operator-1",
    )


def _service_principal(tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="verification-scheduler",
        principal_type=PrincipalType.SERVICE,
        auth_subject="verification-scheduler",
    )


def _build_stack(
    *,
    evaluator: _RecordingEvaluator | None = None,
) -> tuple[
    RuntimeArchitectureGovernanceBridge,
    AdaptationExecutor,
    InMemoryProfileVersionStore,
    InMemoryProfileActivePointerStore,
    _RecordingEvaluator,
]:
    store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store)
    mutation_store = InMemoryAdaptiveProfileMutationStore(
        version_store=store,
        pointer_store=pointer_store,
    )
    approval_store = InMemoryPolicyLearningApprovalStore()
    executor = AdaptationExecutor(
        profile_store=store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
        mutation_store=mutation_store,
        approval_store=approval_store,
    )
    recording = evaluator or _RecordingEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=recording)
    bridge = RuntimeArchitectureGovernanceBridge(mutation_authorization_boundary=boundary)
    return bridge, executor, store, pointer_store, recording


def _shadow_version(executor: AdaptationExecutor, package: AdaptationProposalPackage) -> str:
    return executor.shadow(package, tenant_id=_TENANT, task_class=_TASK_CLASS).candidate_profile_version_id


def test_ahicpm1_apply_allow_executes_with_evidence() -> None:
    bridge, executor, store, pointer_store, recording = _build_stack()
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    result = bridge.apply_approved(
        package,
        executor=executor,
        pointer_store=pointer_store,
        principal=_user_principal(),
        mutation_id="mut-apply-allow",
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        version_id=version_id,
    )
    assert len(recording.calls) == 1
    assert result.apply_result.applied_version_id == _VERSION_ID
    record = store.get(_VERSION_ID)
    assert record is not None
    assert record.status == ProfileVersionStatus.ACTIVE
    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == _VERSION_ID
    assert result.authorization_evidence.mutation_id == "mut-apply-allow"
    assert result.authorization_evidence.policy_action is PolicyAction.ALLOW


def test_ahicpm2_apply_deny_has_zero_writes() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    bridge, executor, store, pointer_store, _recording = _build_stack(evaluator=evaluator)
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    with pytest.raises(AhiGovernanceBlockedError):
        bridge.apply_approved(
            package,
            executor=executor,
            pointer_store=pointer_store,
            principal=_user_principal(),
            mutation_id="mut-apply-deny",
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )
    record = store.get(_VERSION_ID)
    assert record is not None
    assert record.status == ProfileVersionStatus.SHADOW
    assert pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None


def test_ahicpm3_apply_require_human_has_zero_effects() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="needs human"),
    )
    bridge, executor, store, pointer_store, _recording = _build_stack(evaluator=evaluator)
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    with pytest.raises(AhiGovernanceBlockedError) as exc_info:
        bridge.apply_approved(
            package,
            executor=executor,
            pointer_store=pointer_store,
            principal=_user_principal(),
            mutation_id="mut-apply-human",
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )
    assert exc_info.value.authorization_evidence is not None
    assert exc_info.value.authorization_scope is not None
    assert store.get(_VERSION_ID).status == ProfileVersionStatus.SHADOW
    assert pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None


def test_ahicpm4_wrong_tenant_blocked_before_write() -> None:
    bridge, executor, store, pointer_store, recording = _build_stack()
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    with pytest.raises(AhiGovernanceBlockedError) as exc_info:
        bridge.apply_approved(
            package,
            executor=executor,
            pointer_store=pointer_store,
            principal=_user_principal(tenant_id=_OTHER_TENANT),
            mutation_id="mut-wrong-tenant",
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )
    assert exc_info.value.tenant_scope_denial is not None
    assert exc_info.value.authorization_evidence is None
    assert len(recording.calls) == 0


def test_ahicpm5_missing_cpm_dependency_fail_closed() -> None:
    bridge = RuntimeArchitectureGovernanceBridge()
    executor_stack = _build_stack()
    _, executor, _, pointer_store, _ = executor_stack
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    with pytest.raises(AhiGovernanceBlockedError) as exc_info:
        bridge.apply_approved(
            package,
            executor=executor,
            pointer_store=pointer_store,
            principal=_user_principal(),
            mutation_id="mut-missing-boundary",
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )
    assert exc_info.value.blocker_code == "AHI_BLOCKED_BY_MISSING_MUTATION_AUTHORIZATION_BOUNDARY"


def test_ahicpm6_caller_mutation_id_in_evidence() -> None:
    bridge, executor, _, pointer_store, _ = _build_stack()
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    caller_id = "caller-mutation-abc-123"
    result = bridge.apply_approved(
        package,
        executor=executor,
        pointer_store=pointer_store,
        principal=_user_principal(),
        mutation_id=caller_id,
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        version_id=version_id,
    )
    assert result.authorization_evidence.mutation_id == caller_id


def test_ahicpm7_current_target_binding_changes_digest() -> None:
    principal = _user_principal()
    base = build_apply_profile_mutation_request(
        principal=principal,
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        mutation_id="mut-digest",
        artifact_type=ProfileArtifactType.RAG,
        current_active_version_id=None,
        target_version_id=_VERSION_ID,
    )
    changed_current = build_apply_profile_mutation_request(
        principal=principal,
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        mutation_id="mut-digest",
        artifact_type=ProfileArtifactType.RAG,
        current_active_version_id="baseline-v1",
        target_version_id=_VERSION_ID,
    )
    changed_target = build_apply_profile_mutation_request(
        principal=principal,
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        mutation_id="mut-digest",
        artifact_type=ProfileArtifactType.RAG,
        current_active_version_id=None,
        target_version_id="other-version",
    )
    digest_base = control_plane_mutation_request_digest(base)
    assert digest_base != control_plane_mutation_request_digest(changed_current)
    assert digest_base != control_plane_mutation_request_digest(changed_target)


def test_ahicpm8_stale_state_conflict_does_not_overwrite_pointer() -> None:
    _, executor, store, pointer_store, _ = _build_stack()
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="intruder-v",
        expected_active_version_id=None,
    )
    with pytest.raises(ProfileActivePointerConflictError):
        executor.apply(
            package,
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            version_id=version_id,
            expected_active_version_id=None,
        )
    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "intruder-v"
    assert store.get(_VERSION_ID).status == ProfileVersionStatus.SHADOW


def test_ahicpm9_concurrent_cas_sqlite_rejects_lost_update(tmp_path: Path) -> None:
    db_path = tmp_path / "pointers.db"
    store = SQLiteProfileActivePointerStore(db_path=db_path)
    store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="baseline-v1",
        expected_active_version_id=None,
    )
    with pytest.raises(ProfileActivePointerConflictError):
        store.swap_active(
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
            new_active_version_id="lost-update",
            expected_active_version_id=None,
        )
    pointer = store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "baseline-v1"


def _seed_active_pair(
    *,
    store: InMemoryProfileVersionStore,
    pointer_store: InMemoryProfileActivePointerStore,
    lifecycle: ProfileVersionLifecycleManager,
    baseline_id: str = "baseline-v1",
    candidate_id: str = "candidate-v2",
) -> None:
    store.create_from_draft(
        ProfileVersionDraft(
            version_id=baseline_id,
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "standard"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    store.create_from_draft(
        ProfileVersionDraft(
            version_id=candidate_id,
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "deep"},
            parent_version_id=baseline_id,
            status=ProfileVersionStatus.CANARY,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id=baseline_id,
        expected_active_version_id=None,
    )
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id=candidate_id,
        expected_active_version_id=baseline_id,
    )
    lifecycle.transition(baseline_id, target=ProfileVersionStatus.RETIRED)
    lifecycle.transition(candidate_id, target=ProfileVersionStatus.ACTIVE)


def test_ahicpm10_rollback_allow_uses_canonical_previous() -> None:
    bridge, executor, store, pointer_store, recording = _build_stack()
    lifecycle = ProfileVersionLifecycleManager(store)
    _seed_active_pair(store=store, pointer_store=pointer_store, lifecycle=lifecycle)
    result = bridge.rollback_profile(
        executor=executor,
        pointer_store=pointer_store,
        principal=_user_principal(),
        mutation_id="mut-rollback-allow",
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_type == MUTATION_TYPE_ROLLBACK_PROFILE
    assert result.rollback_result.restored_version_id == "baseline-v1"
    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "baseline-v1"


def test_ahicpm11_rollback_deny_zero_effects() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    bridge, executor, store, pointer_store, _ = _build_stack(evaluator=evaluator)
    lifecycle = ProfileVersionLifecycleManager(store)
    _seed_active_pair(store=store, pointer_store=pointer_store, lifecycle=lifecycle)
    with pytest.raises(AhiGovernanceBlockedError):
        bridge.rollback_profile(
            executor=executor,
            pointer_store=pointer_store,
            principal=_user_principal(),
            mutation_id="mut-rollback-deny",
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
        )
    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "candidate-v2"


def test_ahicpm12_rollback_require_human_zero_effects() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="needs human"),
    )
    bridge, executor, store, pointer_store, _ = _build_stack(evaluator=evaluator)
    lifecycle = ProfileVersionLifecycleManager(store)
    _seed_active_pair(store=store, pointer_store=pointer_store, lifecycle=lifecycle)
    with pytest.raises(AhiGovernanceBlockedError) as exc_info:
        bridge.rollback_profile(
            executor=executor,
            pointer_store=pointer_store,
            principal=_user_principal(),
            mutation_id="mut-rollback-human",
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
        )
    assert exc_info.value.authorization_scope is not None
    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer.active_version_id == "candidate-v2"


def test_ahicpm13_rollback_request_binds_canonical_target() -> None:
    principal = _user_principal()
    request = build_rollback_profile_mutation_request(
        principal=principal,
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        mutation_id="mut-rollback-bind",
        artifact_type=ProfileArtifactType.RAG,
        current_active_version_id="candidate-v2",
        target_previous_version_id="baseline-v1",
    )
    assert "baseline-v1" in request.target_revision
    assert request.mutation_type == MUTATION_TYPE_ROLLBACK_PROFILE


def test_ahicpm14_verification_failure_uses_fresh_cpm_evaluation() -> None:
    evaluator = _RecordingEvaluator()
    bridge, executor, store, pointer_store, recording = _build_stack(evaluator=evaluator)
    lifecycle = ProfileVersionLifecycleManager(store)
    _seed_active_pair(store=store, pointer_store=pointer_store, lifecycle=lifecycle)
    signal_store = InMemorySignalStore()
    for utility in (0.30, 0.35, 0.40):
        signal_store.append(
            HarnessOutcomeSignal(
                run_id=f"c-{utility}",
                tenant_id=_TENANT,
                application_id="lab",
                agent_id="agent:echo",
                task_class=_TASK_CLASS,
                eval_mode=OutcomeEvalMode.SHADOW,
                quality_score=utility,
                utility=utility,
            )
        )
    for utility in (0.80, 0.78, 0.79):
        signal_store.append(
            HarnessOutcomeSignal(
                run_id=f"b-{utility}",
                tenant_id=_TENANT,
                application_id="lab",
                agent_id="agent:echo",
                task_class=_TASK_CLASS,
                eval_mode=OutcomeEvalMode.OFFLINE,
                quality_score=utility,
                utility=utility,
            )
        )
    loop = VerificationLoop(
        signal_store=signal_store,
        profile_store=store,
        executor=executor,
        governance_bridge=bridge,
        pointer_store=pointer_store,
        security_checker=_AlwaysPassSecurityChecker(),
    )
    target = VerificationTarget(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        candidate_version_id="candidate-v2",
    )
    result = loop.verify_target(
        target,
        context=VerificationContext(
            min_run_count=3,
            auto_rollback_enabled=True,
            auto_rollback_service_principal=_service_principal(),
            auto_rollback_mutation_id="mut-auto-rollback",
        ),
    )
    assert result.passed is False
    assert result.rolled_back is True
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_type == MUTATION_TYPE_ROLLBACK_PROFILE


class _AlwaysPassSecurityChecker:
    def evaluate(self) -> bool:
        return True


def test_ahicpm15_service_principal_preserved_in_evidence() -> None:
    bridge, executor, store, pointer_store, _ = _build_stack()
    lifecycle = ProfileVersionLifecycleManager(store)
    _seed_active_pair(store=store, pointer_store=pointer_store, lifecycle=lifecycle)
    service = _service_principal()
    result = bridge.rollback_profile(
        executor=executor,
        pointer_store=pointer_store,
        principal=service,
        mutation_id="mut-service-rollback",
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert result.authorization_evidence.principal_type is PrincipalType.SERVICE
    assert result.authorization_evidence.principal_auth_subject == service.auth_subject


def test_ahicpm16_separate_rollback_mutation_id() -> None:
    bridge, executor, store, pointer_store, _ = _build_stack()
    lifecycle = ProfileVersionLifecycleManager(store)
    _seed_active_pair(store=store, pointer_store=pointer_store, lifecycle=lifecycle)
    rollback_id = "rollback-mutation-unique-99"
    result = bridge.rollback_profile(
        executor=executor,
        pointer_store=pointer_store,
        principal=_user_principal(),
        mutation_id=rollback_id,
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert result.authorization_evidence.mutation_id == rollback_id


def test_ahicpm17_production_roots_use_governed_facade_only() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    production_roots = [
        repo_root / "intergrax" / "applications",
        repo_root / "intergrax" / "runtime" / "adaptive",
        repo_root / "intergrax" / "runtime" / "architecture",
    ]
    direct_apply: list[str] = []
    direct_rollback: list[str] = []
    allowed_internal = {
        repo_root / "intergrax" / "runtime" / "adaptive" / "adaptation_executor.py",
        repo_root / "intergrax" / "runtime" / "architecture" / "runtime_governance_bridge.py",
    }
    for root in production_roots:
        for path in root.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8")
            if ".apply(" in text and "executor.apply(" in text and path not in allowed_internal:
                direct_apply.append(str(path.relative_to(repo_root)))
            if "executor.rollback(" in text and path not in allowed_internal:
                direct_rollback.append(str(path.relative_to(repo_root)))
    assert direct_apply == []
    assert direct_rollback == []


def test_ahicpm18_single_cpm_evaluation_per_mutation() -> None:
    evaluator = _RecordingEvaluator()
    bridge, executor, store, pointer_store, recording = _build_stack(evaluator=evaluator)
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    bridge.apply_approved(
        package,
        executor=executor,
        pointer_store=pointer_store,
        principal=_user_principal(),
        mutation_id="mut-single-eval",
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        version_id=version_id,
    )
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_type == MUTATION_TYPE_APPLY_PROFILE


def test_ahicpm19_no_dynamic_access_in_governance_slice() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    slice_paths = [
        repo_root / "intergrax" / "runtime" / "adaptive" / "control_plane_governance.py",
        repo_root / "intergrax" / "runtime" / "architecture" / "runtime_governance_bridge.py",
    ]
    forbidden = ("getattr(", "setattr(", "hasattr(", "__dict__", "eval(", "exec(")
    for path in slice_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_ahicpm20_real_evidence_only() -> None:
    bridge, executor, _, pointer_store, _ = _build_stack()
    package = _routing_package()
    version_id = _shadow_version(executor, package)
    result = bridge.apply_approved(
        package,
        executor=executor,
        pointer_store=pointer_store,
        principal=_user_principal(),
        mutation_id="real-mutation-id",
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        version_id=version_id,
    )
    evidence = result.authorization_evidence
    assert evidence.mutation_id == "real-mutation-id"
    assert evidence.request_digest.startswith("sha256:")
    assert evidence.request_digest != "sha256:"
    assert evidence.mutation_type == MUTATION_TYPE_APPLY_PROFILE


def _seed_sqlite_apply_race_state(
    db_path: Path,
    *,
    baseline_id: str = "v10",
    candidate_id: str = "v11",
    intruder_id: str = "v12",
) -> SQLiteAdaptiveProfileMutationStore:
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id=baseline_id,
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "baseline"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id=candidate_id,
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "candidate"},
            parent_version_id=baseline_id,
            status=ProfileVersionStatus.CANARY,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id=intruder_id,
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "intruder"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id=baseline_id,
        expected_active_version_id=None,
    )
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id=intruder_id,
        expected_active_version_id=baseline_id,
    )
    return SQLiteAdaptiveProfileMutationStore(db_path=db_path)


def test_ahicpm21_apply_mid_flight_conflict_zero_partial_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "adaptive_harness.db"
    mutation_store = _seed_sqlite_apply_race_state(db_path)
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)

    with pytest.raises(ProfileActivePointerConflictError):
        mutation_store.commit_apply(
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
            target_version_id="v11",
            expected_active_version_id="v10",
        )

    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "v12"
    assert version_store.get("v11").status == ProfileVersionStatus.CANARY
    assert version_store.get("v10").status == ProfileVersionStatus.ACTIVE


def test_ahicpm22_rollback_mid_flight_conflict_zero_partial_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "adaptive_harness.db"
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="baseline-v1",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "baseline"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="candidate-v2",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "candidate"},
            parent_version_id="baseline-v1",
            status=ProfileVersionStatus.CANARY,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="intruder-v3",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "intruder"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="baseline-v1",
        expected_active_version_id=None,
    )
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="candidate-v2",
        expected_active_version_id="baseline-v1",
    )
    lifecycle = ProfileVersionLifecycleManager(version_store)
    lifecycle.transition("baseline-v1", target=ProfileVersionStatus.RETIRED)
    lifecycle.transition("candidate-v2", target=ProfileVersionStatus.ACTIVE)
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="intruder-v3",
        expected_active_version_id="candidate-v2",
    )
    mutation_store = SQLiteAdaptiveProfileMutationStore(db_path=db_path)

    with pytest.raises(ProfileActivePointerConflictError):
        mutation_store.commit_rollback(
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
            expected_active_version_id="candidate-v2",
        )

    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "intruder-v3"
    assert version_store.get("candidate-v2").status == ProfileVersionStatus.ACTIVE
    assert version_store.get("baseline-v1").status == ProfileVersionStatus.RETIRED


def test_ahicpm23_successful_apply_atomic_commit(tmp_path: Path) -> None:
    db_path = tmp_path / "adaptive_harness.db"
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="v10",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "baseline"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="v11",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "candidate"},
            parent_version_id="v10",
            status=ProfileVersionStatus.CANARY,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="v10",
        expected_active_version_id=None,
    )
    mutation_store = SQLiteAdaptiveProfileMutationStore(db_path=db_path)
    pointer = mutation_store.commit_apply(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        target_version_id="v11",
        expected_active_version_id="v10",
    )
    assert pointer.active_version_id == "v11"
    assert pointer.previous_version_id == "v10"
    assert version_store.get("v11").status == ProfileVersionStatus.ACTIVE
    assert version_store.get("v10").status == ProfileVersionStatus.RETIRED


def test_ahicpm24_successful_rollback_atomic_commit(tmp_path: Path) -> None:
    db_path = tmp_path / "adaptive_harness.db"
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="baseline-v1",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "baseline"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="candidate-v2",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "candidate"},
            parent_version_id="baseline-v1",
            status=ProfileVersionStatus.CANARY,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="baseline-v1",
        expected_active_version_id=None,
    )
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="candidate-v2",
        expected_active_version_id="baseline-v1",
    )
    lifecycle = ProfileVersionLifecycleManager(version_store)
    lifecycle.transition("baseline-v1", target=ProfileVersionStatus.RETIRED)
    lifecycle.transition("candidate-v2", target=ProfileVersionStatus.ACTIVE)

    mutation_store = SQLiteAdaptiveProfileMutationStore(db_path=db_path)
    pointer = mutation_store.commit_rollback(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        expected_active_version_id="candidate-v2",
    )
    assert pointer.active_version_id == "baseline-v1"
    assert pointer.previous_version_id == "candidate-v2"
    assert version_store.get("baseline-v1").status == ProfileVersionStatus.ACTIVE
    assert version_store.get("candidate-v2").status == ProfileVersionStatus.DRAFT


def test_ahicpm25_durable_sqlite_atomicity_rolls_back_invalid_transition(tmp_path: Path) -> None:
    db_path = tmp_path / "adaptive_harness.db"
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="v10",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "baseline"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id="v11",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "draft"},
            status=ProfileVersionStatus.DRAFT,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="v10",
        expected_active_version_id=None,
    )
    mutation_store = SQLiteAdaptiveProfileMutationStore(db_path=db_path)

    with pytest.raises(ProfileLifecycleTransitionError):
        mutation_store.commit_apply(
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
            target_version_id="v11",
            expected_active_version_id="v10",
        )

    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "v10"
    assert version_store.get("v10").status == ProfileVersionStatus.ACTIVE
    assert version_store.get("v11").status == ProfileVersionStatus.DRAFT
