# © Artur Czarnecki. All rights reserved.

"""AHI-PROMOTION-AUTHORITY-INTEGRITY: executor governance and scope-bound promotion tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
)
from intergrax.runtime.adaptive.contracts import (
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.policy_learning_approval import InMemoryPolicyLearningApprovalStore
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

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT_A = "tenant_a"
_TENANT_B = "tenant_b"
_TASK_CLASS = "echo.basic"
_VERSION_ID = "draft-routing-echo-rag_tier_deep"
_PROPOSAL_ID = "prop_routing_01"


def _routing_package(
    *,
    passed_all_gates: bool = True,
    proposal_id: str = _PROPOSAL_ID,
    version_id: str = _VERSION_ID,
) -> AdaptationProposalPackage:
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
        proposal_id=proposal_id,
        candidate=AdaptationProposalCandidate(
            loop_id=envelope.loop_id,
            source_engine="routing_tuning",
            proposal=AdaptiveLoopProposal(
                envelope=envelope,
                proposed_change_summary="Recommend routing shift toward rag_tier_deep",
            ),
            profile_draft=ProfileVersionDraft(
                version_id=version_id,
                artifact_type=ProfileArtifactType.RAG,
                artifact_payload={"selected_arm": "rag_tier_deep"},
            ),
        ),
        envelope_gate=AdaptiveLoopGateResult(loop_id=envelope.loop_id, passed=True),
        passed_all_gates=passed_all_gates,
    )


def _build_executor() -> tuple[
    AdaptationExecutor,
    InMemoryProfileVersionStore,
    InMemoryProfileActivePointerStore,
]:
    store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store)
    executor = AdaptationExecutor(
        profile_store=store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
        approval_store=InMemoryPolicyLearningApprovalStore(),
    )
    return executor, store, pointer_store


def _shadow_version(
    executor: AdaptationExecutor,
    package: AdaptationProposalPackage,
    *,
    tenant_id: str = _TENANT_A,
    task_class: str = _TASK_CLASS,
) -> str:
    result = executor.shadow(package, tenant_id=tenant_id, task_class=task_class)
    return result.candidate_profile_version_id


def test_apply_rejects_failed_governance_gates() -> None:
    executor, store, pointer_store = _build_executor()
    package = _routing_package(passed_all_gates=False)
    failed_package = package.model_copy(update={"passed_all_gates": False})
    passing_package = _routing_package()
    version_id = _shadow_version(executor, passing_package)

    with pytest.raises(ValueError, match="failed governance gates"):
        executor.apply(
            failed_package,
            tenant_id=_TENANT_A,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )

    record = store.get(version_id)
    assert record is not None
    assert record.status == ProfileVersionStatus.SHADOW
    assert pointer_store.get_pointer(
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None


def test_apply_rejects_package_version_mismatch() -> None:
    executor, store, pointer_store = _build_executor()
    package_v1 = _routing_package(version_id="draft-v1")
    version_v1 = _shadow_version(executor, package_v1)
    package_v2 = _routing_package(version_id="draft-v2", proposal_id="prop_routing_02")
    _shadow_version(executor, package_v2)

    with pytest.raises(ValueError, match="version_id mismatch"):
        executor.apply(
            package_v1,
            tenant_id=_TENANT_A,
            task_class=_TASK_CLASS,
            version_id="draft-v2",
        )

    assert store.get("draft-v2") is not None
    assert store.get("draft-v2").status == ProfileVersionStatus.SHADOW
    assert store.get(version_v1) is not None
    assert store.get(version_v1).status == ProfileVersionStatus.SHADOW
    assert pointer_store.get_pointer(
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None


def test_apply_rejects_proposal_lineage_mismatch() -> None:
    executor, store, pointer_store = _build_executor()
    store.create_from_draft(
        ProfileVersionDraft(
            version_id=_VERSION_ID,
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"selected_arm": "rag_tier_deep"},
            created_by="prop_unrelated",
        ),
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
    )
    lifecycle = ProfileVersionLifecycleManager(store)
    lifecycle.transition(_VERSION_ID, target=ProfileVersionStatus.SHADOW)
    package = _routing_package()

    with pytest.raises(ValueError, match="lineage mismatch"):
        executor.apply(
            package,
            tenant_id=_TENANT_A,
            task_class=_TASK_CLASS,
            version_id=_VERSION_ID,
        )

    record = store.get(_VERSION_ID)
    assert record is not None
    assert record.status == ProfileVersionStatus.SHADOW
    assert pointer_store.get_pointer(
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None


def test_apply_rejects_cross_tenant_scope() -> None:
    executor, store, pointer_store = _build_executor()
    package = _routing_package()
    version_id = _shadow_version(executor, package, tenant_id=_TENANT_B, task_class=_TASK_CLASS)

    with pytest.raises(ValueError, match="tenant mismatch"):
        executor.apply(
            package,
            tenant_id=_TENANT_A,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )

    record = store.get(version_id)
    assert record is not None
    assert record.status == ProfileVersionStatus.SHADOW
    assert record.tenant_id == _TENANT_B
    assert pointer_store.get_pointer(
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None
    assert pointer_store.get_pointer(
        tenant_id=_TENANT_B,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None


def test_apply_rejects_task_class_mismatch() -> None:
    executor, store, pointer_store = _build_executor()
    package = _routing_package()
    version_id = _shadow_version(executor, package, tenant_id=_TENANT_A, task_class="echo.other")

    with pytest.raises(ValueError, match="task_class mismatch"):
        executor.apply(
            package,
            tenant_id=_TENANT_A,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )

    record = store.get(version_id)
    assert record is not None
    assert record.status == ProfileVersionStatus.SHADOW
    assert record.task_class == "echo.other"
    assert pointer_store.get_pointer(
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    ) is None


def test_canary_rejects_cross_tenant_scope() -> None:
    executor, store, _pointer_store = _build_executor()
    package = _routing_package()
    version_id = _shadow_version(executor, package, tenant_id=_TENANT_B, task_class=_TASK_CLASS)

    with pytest.raises(ValueError, match="tenant mismatch"):
        executor.canary(
            tenant_id=_TENANT_A,
            task_class=_TASK_CLASS,
            version_id=version_id,
        )

    record = store.get(version_id)
    assert record is not None
    assert record.status == ProfileVersionStatus.SHADOW


def test_valid_matching_apply_succeeds() -> None:
    executor, store, pointer_store = _build_executor()
    package = _routing_package()
    version_id = _shadow_version(executor, package)

    result = executor.apply(
        package,
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        version_id=version_id,
    )

    assert result.applied_version_id == _VERSION_ID
    record = store.get(_VERSION_ID)
    assert record is not None
    assert record.status == ProfileVersionStatus.ACTIVE
    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == _VERSION_ID


def test_valid_canary_succeeds() -> None:
    executor, store, _pointer_store = _build_executor()
    package = _routing_package()
    version_id = _shadow_version(executor, package)

    result = executor.canary(
        tenant_id=_TENANT_A,
        task_class=_TASK_CLASS,
        version_id=version_id,
    )

    assert result.version_id == _VERSION_ID
    record = store.get(version_id)
    assert record is not None
    assert record.status == ProfileVersionStatus.CANARY
