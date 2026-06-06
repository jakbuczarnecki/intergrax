# © Artur Czarnecki. All rights reserved.

"""Adaptation executor for shadow allocation (Phase W-ADAPT-3.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.adaptation_models import AdaptationProposalPackage
from intergrax.runtime.adaptive.contracts import ProfileVersionDraft, ProfileVersionStatus
from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
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


class AdaptationExecutor:
    """
    Materialize approved proposals as profile versions.

    Wave 3 scope: ``shadow()`` only — no canary/apply/rollback yet.
    """

    def __init__(
        self,
        *,
        profile_store: ProfileVersionStore,
        lifecycle_manager: ProfileVersionLifecycleManager,
    ) -> None:
        self._profile_store = profile_store
        self._lifecycle_manager = lifecycle_manager

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
