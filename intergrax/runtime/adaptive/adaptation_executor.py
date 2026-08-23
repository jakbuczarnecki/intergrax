# © Artur Czarnecki. All rights reserved.

"""Adaptation executor for shadow/canary/apply/rollback (Phase W-ADAPT-3.3, W-ADAPT-4.4–4.5)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.adaptation_models import AdaptationProposalPackage
from intergrax.runtime.adaptive.contracts import (
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionRecord,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.policy_learning_approval import (
    PolicyLearningApprovalStore,
    require_policy_learning_approval,
)
from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
from intergrax.runtime.adaptive.profile_pointer_store import (
    ProfileActivePointerConflictError,
    ProfileActivePointerStore,
)
from intergrax.runtime.adaptive.profile_version_store import ProfileVersionStore


class ShadowAllocationResult(BaseModel):
    """Outcome of allocating a governed proposal for shadow evaluation."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    candidate_profile_version_id: str
    proposal_id: str
    loop_id: str
    trace_tag: str = Field(
        default="candidate_profile_version_id",
        description="Trace metadata key for shadow runs",
    )


class CanaryPromotionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    version_id: str
    artifact_type: ProfileArtifactType


class ApplyProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    applied_version_id: str
    artifact_type: ProfileArtifactType
    previous_version_id: str | None = None


class RollbackProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    restored_version_id: str
    artifact_type: ProfileArtifactType
    rolled_back_version_id: str | None = None


class AdaptationExecutor:
    """Materialize approved proposals and shift active profile version pointers."""

    def __init__(
        self,
        *,
        profile_store: ProfileVersionStore,
        pointer_store: ProfileActivePointerStore,
        lifecycle_manager: ProfileVersionLifecycleManager,
        approval_store: PolicyLearningApprovalStore | None = None,
    ) -> None:
        self._profile_store = profile_store
        self._pointer_store = pointer_store
        self._lifecycle_manager = lifecycle_manager
        self._approval_store = approval_store

    def shadow(
        self,
        package: AdaptationProposalPackage,
        *,
        tenant_id: str,
        task_class: str,
    ) -> ShadowAllocationResult:
        if not package.passed_all_gates:
            raise ValueError("Cannot shadow a proposal that failed governance gates")
        draft = package.candidate.profile_draft
        if draft is None:
            raise ValueError("Shadow allocation requires a ProfileVersionDraft")

        record = self._profile_store.create_from_draft(
            ProfileVersionDraft(
                version_id=draft.version_id,
                artifact_type=draft.artifact_type,
                artifact_payload=dict(draft.artifact_payload),
                parent_version_id=draft.parent_version_id,
                created_by=package.proposal_id,
                status=ProfileVersionStatus.DRAFT,
            ),
            tenant_id=tenant_id,
            task_class=task_class,
        )
        promoted = self._lifecycle_manager.transition(
            record.version_id,
            target=ProfileVersionStatus.SHADOW,
        )
        return ShadowAllocationResult(
            tenant_id=tenant_id,
            task_class=task_class,
            candidate_profile_version_id=promoted.version_id,
            proposal_id=package.proposal_id,
            loop_id=package.candidate.loop_id,
        )

    def canary(
        self,
        *,
        tenant_id: str,
        task_class: str,
        version_id: str,
    ) -> CanaryPromotionResult:
        record = self._require_scoped_version(
            version_id,
            tenant_id=tenant_id,
            task_class=task_class,
        )
        if record.status != ProfileVersionStatus.SHADOW:
            raise ValueError(f"Canary promotion requires shadow status, got {record.status.value}")
        promoted = self._lifecycle_manager.transition(version_id, target=ProfileVersionStatus.CANARY)
        return CanaryPromotionResult(
            tenant_id=tenant_id,
            task_class=task_class,
            version_id=promoted.version_id,
            artifact_type=promoted.artifact_type,
        )

    def apply(
        self,
        package: AdaptationProposalPackage,
        *,
        tenant_id: str,
        task_class: str,
        version_id: str,
        expected_active_version_id: str | None,
    ) -> ApplyProfileResult:
        record = self._validate_apply_promotion_authority(
            package,
            tenant_id=tenant_id,
            task_class=task_class,
            version_id=version_id,
        )
        if self._approval_store is not None:
            require_policy_learning_approval(package, approval_store=self._approval_store)
        if record.status not in {ProfileVersionStatus.SHADOW, ProfileVersionStatus.CANARY}:
            raise ValueError(
                f"Apply requires shadow or canary status, got {record.status.value}"
            )
        self._assert_pointer_matches_expected(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=record.artifact_type,
            expected_active_version_id=expected_active_version_id,
        )
        if record.status == ProfileVersionStatus.SHADOW:
            record = self._lifecycle_manager.transition(version_id, target=ProfileVersionStatus.CANARY)
        active = self._lifecycle_manager.transition(record.version_id, target=ProfileVersionStatus.ACTIVE)

        pointer = self._pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=active.artifact_type,
        )
        if pointer is not None and pointer.active_version_id != active.version_id:
            previous = self._profile_store.get(pointer.active_version_id)
            if previous is not None and previous.status == ProfileVersionStatus.ACTIVE:
                self._lifecycle_manager.transition(
                    previous.version_id,
                    target=ProfileVersionStatus.RETIRED,
                )

        swapped = self._pointer_store.swap_active(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=active.artifact_type,
            new_active_version_id=active.version_id,
            expected_active_version_id=expected_active_version_id,
        )
        return ApplyProfileResult(
            tenant_id=tenant_id,
            task_class=task_class,
            applied_version_id=active.version_id,
            artifact_type=active.artifact_type,
            previous_version_id=swapped.previous_version_id,
        )

    def rollback(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        expected_active_version_id: str,
    ) -> RollbackProfileResult:
        pointer = self._pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        if pointer is None or pointer.previous_version_id is None:
            raise ValueError("No rollback pointer available for active profile version")
        if pointer.active_version_id != expected_active_version_id:
            raise ProfileActivePointerConflictError(
                "active profile pointer changed before rollback"
            )

        current = self._profile_store.get(pointer.active_version_id)
        previous = self._profile_store.get(pointer.previous_version_id)
        if current is None or previous is None:
            raise ValueError("Rollback requires both current and previous profile versions")

        if current.status == ProfileVersionStatus.ACTIVE:
            self._lifecycle_manager.transition(current.version_id, target=ProfileVersionStatus.DRAFT)
        restored = self._lifecycle_manager.transition(
            previous.version_id,
            target=ProfileVersionStatus.ACTIVE,
        )
        self._pointer_store.swap_active(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            new_active_version_id=restored.version_id,
            expected_active_version_id=expected_active_version_id,
        )
        return RollbackProfileResult(
            tenant_id=tenant_id,
            task_class=task_class,
            restored_version_id=restored.version_id,
            artifact_type=artifact_type,
            rolled_back_version_id=current.version_id,
        )

    def _require_scoped_version(
        self,
        version_id: str,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType | None = None,
    ) -> ProfileVersionRecord:
        record = self._profile_store.get(version_id)
        if record is None:
            raise ValueError(f"Unknown profile version: {version_id}")
        if record.tenant_id != tenant_id:
            raise ValueError(
                "Profile version tenant mismatch: "
                f"expected {tenant_id!r}, got {record.tenant_id!r}"
            )
        if record.task_class != task_class:
            raise ValueError(
                "Profile version task_class mismatch: "
                f"expected {task_class!r}, got {record.task_class!r}"
            )
        if artifact_type is not None and record.artifact_type != artifact_type:
            raise ValueError(
                "Profile version artifact_type mismatch: "
                f"expected {artifact_type.value!r}, got {record.artifact_type.value!r}"
            )
        return record

    def _assert_pointer_matches_expected(
        self,
        *,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
        expected_active_version_id: str | None,
    ) -> None:
        pointer = self._pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        actual_active = pointer.active_version_id if pointer is not None else None
        if actual_active != expected_active_version_id:
            raise ProfileActivePointerConflictError(
                "active profile pointer changed before mutation"
            )

    def _validate_apply_promotion_authority(
        self,
        package: AdaptationProposalPackage,
        *,
        tenant_id: str,
        task_class: str,
        version_id: str,
    ) -> ProfileVersionRecord:
        if not package.passed_all_gates:
            raise ValueError("Cannot apply a proposal that failed governance gates")
        draft = package.candidate.profile_draft
        if draft is None:
            raise ValueError("Apply requires a ProfileVersionDraft")
        if version_id != draft.version_id:
            raise ValueError(
                "Apply version_id mismatch: "
                f"package authorizes {draft.version_id!r}, got {version_id!r}"
            )
        record = self._require_scoped_version(
            version_id,
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=draft.artifact_type,
        )
        if record.created_by != package.proposal_id:
            raise ValueError(
                "Profile version lineage mismatch: "
                f"created_by {record.created_by!r} != proposal {package.proposal_id!r}"
            )
        return record
