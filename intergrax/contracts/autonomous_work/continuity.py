# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""WorkContinuityState orientation contract (AW-1A)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
)
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerGoalId,
    WorkerInstanceId,
    validate_responsibility_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import (
    ArtifactReference,
    ContextAnchorReference,
    ExternalDependencyReference,
    HumanPendingReference,
    ProblemReference,
    ProgressCheckpointRef,
    WorkReference,
    validate_artifact_reference,
    validate_context_anchor_reference,
    validate_external_dependency_reference,
    validate_human_pending_reference,
    validate_problem_reference,
    validate_progress_checkpoint_ref,
    validate_work_reference,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision
from intergrax.contracts.decision_identity import DecisionId, validate_decision_id


@dataclass(frozen=True, slots=True)
class ProgressCheckpoint:
    """Latest durable progress marker — not a conversation or execution trace."""

    checkpointed_at: datetime
    summary_ref: ProgressCheckpointRef | None = None

    def __post_init__(self) -> None:
        require_aware_utc(self.checkpointed_at, label="checkpointed_at")
        if self.summary_ref is not None:
            validate_progress_checkpoint_ref(self.summary_ref)


@dataclass(frozen=True, slots=True)
class WorkContinuityState:
    """Durable worker orientation snapshot — not memory, chat history, or RAG index."""

    worker_instance_ref: WorkerInstanceId
    responsibility_refs: tuple[ResponsibilityId, ...]
    active_goal_refs: tuple[WorkerGoalId, ...]
    open_work_refs: tuple[WorkReference, ...]
    blocked_work_refs: tuple[WorkReference, ...]
    pending_external_refs: tuple[ExternalDependencyReference, ...]
    pending_human_refs: tuple[HumanPendingReference, ...]
    recent_decision_refs: tuple[DecisionId, ...]
    relevant_artifact_refs: tuple[ArtifactReference, ...]
    unresolved_problem_refs: tuple[ProblemReference, ...]
    last_progress_checkpoint: ProgressCheckpoint | None
    next_action_hint: str | None
    context_anchor_refs: tuple[ContextAnchorReference, ...]
    revision: Revision

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_ref)
        object.__setattr__(
            self,
            "responsibility_refs",
            freeze_tuple(self.responsibility_refs, label="responsibility_refs"),
        )
        object.__setattr__(
            self,
            "active_goal_refs",
            freeze_tuple(self.active_goal_refs, label="active_goal_refs"),
        )
        object.__setattr__(
            self,
            "open_work_refs",
            freeze_tuple(self.open_work_refs, label="open_work_refs"),
        )
        object.__setattr__(
            self,
            "blocked_work_refs",
            freeze_tuple(self.blocked_work_refs, label="blocked_work_refs"),
        )
        object.__setattr__(
            self,
            "pending_external_refs",
            freeze_tuple(self.pending_external_refs, label="pending_external_refs"),
        )
        object.__setattr__(
            self,
            "pending_human_refs",
            freeze_tuple(self.pending_human_refs, label="pending_human_refs"),
        )
        object.__setattr__(
            self,
            "recent_decision_refs",
            freeze_tuple(self.recent_decision_refs, label="recent_decision_refs"),
        )
        object.__setattr__(
            self,
            "relevant_artifact_refs",
            freeze_tuple(self.relevant_artifact_refs, label="relevant_artifact_refs"),
        )
        object.__setattr__(
            self,
            "unresolved_problem_refs",
            freeze_tuple(self.unresolved_problem_refs, label="unresolved_problem_refs"),
        )
        object.__setattr__(
            self,
            "context_anchor_refs",
            freeze_tuple(self.context_anchor_refs, label="context_anchor_refs"),
        )
        for index, responsibility_id in enumerate(self.responsibility_refs):
            validate_responsibility_id(responsibility_id)
        for index, goal_id in enumerate(self.active_goal_refs):
            validate_worker_goal_id(goal_id)
        for index, work_ref in enumerate(self.open_work_refs):
            validate_work_reference(work_ref)
        for index, work_ref in enumerate(self.blocked_work_refs):
            validate_work_reference(work_ref)
        for index, external_ref in enumerate(self.pending_external_refs):
            validate_external_dependency_reference(external_ref)
        for index, human_ref in enumerate(self.pending_human_refs):
            validate_human_pending_reference(human_ref)
        for index, decision_id in enumerate(self.recent_decision_refs):
            validate_decision_id(decision_id)
        for index, artifact_ref in enumerate(self.relevant_artifact_refs):
            validate_artifact_reference(artifact_ref)
        for index, problem_ref in enumerate(self.unresolved_problem_refs):
            validate_problem_reference(problem_ref)
        for index, anchor_ref in enumerate(self.context_anchor_refs):
            validate_context_anchor_reference(anchor_ref)
        if self.last_progress_checkpoint is not None:
            if type(self.last_progress_checkpoint) is not ProgressCheckpoint:
                raise TypeError("last_progress_checkpoint must be ProgressCheckpoint")
        if self.next_action_hint is not None:
            if type(self.next_action_hint) is not str:
                raise TypeError("next_action_hint must be str")
            if not self.next_action_hint.strip():
                raise ValueError("next_action_hint must be non-empty when provided")
        if type(self.revision) is not Revision:
            raise TypeError("revision must be Revision")
        validate_revision(self.revision)
