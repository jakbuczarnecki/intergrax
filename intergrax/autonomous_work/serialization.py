# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic JSON codec for durable Autonomous Work records (AW-2C)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, TypeVar

from intergrax.contracts.autonomous_work.continuity import (
    ProgressCheckpoint,
    WorkContinuityState,
)
from intergrax.contracts.autonomous_work.goal import WorkerGoal, WorkerGoalStatus
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerDefinitionId,
    WorkerGoalId,
    WorkerInstanceId,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.principal_binding import WorkerPrincipalBinding
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    CapabilityProfileRef,
    CodecraftProfileRef,
    CollaborationProfileRef,
    EscalationPolicyRef,
    GovernanceProfileRef,
    MemoryProfileRef,
    ObservabilityProfileRef,
    ProfileVersion,
    RiskProfileRef,
    ScheduleProfileRef,
)
from intergrax.contracts.autonomous_work.references import (
    ArtifactReference,
    ContextAnchorReference,
    DeadlineOrCadenceRef,
    DefaultGoalPolicyRef,
    EvaluationCadenceRef,
    ExternalDependencyReference,
    HumanPendingReference,
    MetricRef,
    PrincipalBindingPolicyRef,
    PrincipalBindingRef,
    ProblemReference,
    ProgressCheckpointRef,
    ProgressProjectionRef,
    ResponsibilityScopeRef,
    ResponsibilityTemplateRef,
    SlaSloRef,
    SuccessCriteriaRef,
    WorkReference,
    WorkspaceContextRef,
    WorkspaceScopeRef,
)
from intergrax.contracts.autonomous_work.responsibility import (
    Responsibility,
    ResponsibilityStatus,
)
from intergrax.contracts.autonomous_work.revision import DefinitionRevision, Revision
from intergrax.contracts.autonomous_work.worker import WorkerDefinition, WorkerInstance
from intergrax.contracts.decision_identity import DecisionId

CODEC_VERSION = 1

_TProfileRef = TypeVar(
    "_TProfileRef",
    GovernanceProfileRef,
    BudgetProfileRef,
    MemoryProfileRef,
    CapabilityProfileRef,
    CodecraftProfileRef,
    RiskProfileRef,
    ScheduleProfileRef,
    EscalationPolicyRef,
    CollaborationProfileRef,
    ObservabilityProfileRef,
)


def stable_record_json(payload: dict[str, Any]) -> str:
    """Return canonical JSON for durable storage and content comparison."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _encode_datetime(value: datetime) -> str:
    return value.isoformat()


def _decode_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _encode_profile_ref(ref: _TProfileRef) -> dict[str, Any]:
    return {"profile_id": ref.profile_id, "version": ref.version.value}


def _decode_profile_ref(
    payload: dict[str, Any],
    ref_type: Callable[[str, ProfileVersion], _TProfileRef],
) -> _TProfileRef:
    return ref_type(payload["profile_id"], ProfileVersion(payload["version"]))


def _encode_string_tuple(values: tuple[str, ...]) -> list[str]:
    return list(values)


def worker_definition_to_payload(definition: WorkerDefinition) -> dict[str, Any]:
    return {
        "codec_version": CODEC_VERSION,
        "worker_definition_id": definition.worker_definition_id,
        "display_name": definition.display_name,
        "role": definition.role,
        "revision": definition.revision.value,
        "responsibility_template_refs": _encode_string_tuple(
            definition.responsibility_template_refs
        ),
        "default_goal_policy_ref": definition.default_goal_policy_ref,
        "principal_binding_policy_ref": definition.principal_binding_policy_ref,
        "workspace_scope_ref": definition.workspace_scope_ref,
        "governance_profile_ref": _encode_profile_ref(definition.governance_profile_ref),
        "budget_profile_ref": _encode_profile_ref(definition.budget_profile_ref),
        "memory_profile_ref": _encode_profile_ref(definition.memory_profile_ref),
        "capability_profile_ref": _encode_profile_ref(definition.capability_profile_ref),
        "codecraft_profile_ref": _encode_profile_ref(definition.codecraft_profile_ref),
        "risk_profile_ref": _encode_profile_ref(definition.risk_profile_ref),
        "schedule_profile_ref": _encode_profile_ref(definition.schedule_profile_ref),
        "escalation_policy_ref": _encode_profile_ref(definition.escalation_policy_ref),
        "collaboration_profile_ref": _encode_profile_ref(
            definition.collaboration_profile_ref
        ),
        "observability_profile_ref": _encode_profile_ref(
            definition.observability_profile_ref
        ),
    }


def worker_definition_from_payload(payload: dict[str, Any]) -> WorkerDefinition:
    if payload.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported WorkerDefinition codec version")
    return WorkerDefinition(
        worker_definition_id=WorkerDefinitionId(payload["worker_definition_id"]),
        display_name=payload["display_name"],
        role=payload["role"],
        revision=DefinitionRevision(payload["revision"]),
        responsibility_template_refs=tuple(
            ResponsibilityTemplateRef(value)
            for value in payload["responsibility_template_refs"]
        ),
        default_goal_policy_ref=DefaultGoalPolicyRef(payload["default_goal_policy_ref"]),
        principal_binding_policy_ref=PrincipalBindingPolicyRef(
            payload["principal_binding_policy_ref"]
        ),
        workspace_scope_ref=WorkspaceScopeRef(payload["workspace_scope_ref"]),
        governance_profile_ref=_decode_profile_ref(
            payload["governance_profile_ref"], GovernanceProfileRef
        ),
        budget_profile_ref=_decode_profile_ref(
            payload["budget_profile_ref"], BudgetProfileRef
        ),
        memory_profile_ref=_decode_profile_ref(
            payload["memory_profile_ref"], MemoryProfileRef
        ),
        capability_profile_ref=_decode_profile_ref(
            payload["capability_profile_ref"], CapabilityProfileRef
        ),
        codecraft_profile_ref=_decode_profile_ref(
            payload["codecraft_profile_ref"], CodecraftProfileRef
        ),
        risk_profile_ref=_decode_profile_ref(payload["risk_profile_ref"], RiskProfileRef),
        schedule_profile_ref=_decode_profile_ref(
            payload["schedule_profile_ref"], ScheduleProfileRef
        ),
        escalation_policy_ref=_decode_profile_ref(
            payload["escalation_policy_ref"], EscalationPolicyRef
        ),
        collaboration_profile_ref=_decode_profile_ref(
            payload["collaboration_profile_ref"], CollaborationProfileRef
        ),
        observability_profile_ref=_decode_profile_ref(
            payload["observability_profile_ref"], ObservabilityProfileRef
        ),
    )


def worker_definition_to_json(definition: WorkerDefinition) -> str:
    return stable_record_json(worker_definition_to_payload(definition))


def worker_definition_from_json(payload: str) -> WorkerDefinition:
    return worker_definition_from_payload(json.loads(payload))


def worker_instance_to_payload(instance: WorkerInstance) -> dict[str, Any]:
    return {
        "codec_version": CODEC_VERSION,
        "worker_instance_id": instance.worker_instance_id,
        "worker_definition_id": instance.worker_definition_id,
        "definition_revision": instance.definition_revision.value,
        "lifecycle_state": instance.lifecycle_state.value,
        "principal_binding_ref": instance.principal_binding_ref,
        "workspace_context_ref": instance.workspace_context_ref,
        "active_responsibility_refs": _encode_string_tuple(
            instance.active_responsibility_refs
        ),
        "active_goal_refs": _encode_string_tuple(instance.active_goal_refs),
        "created_at": _encode_datetime(instance.created_at),
        "updated_at": _encode_datetime(instance.updated_at),
        "revision": instance.revision.value,
    }


def worker_instance_from_payload(payload: dict[str, Any]) -> WorkerInstance:
    if payload.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported WorkerInstance codec version")
    return WorkerInstance(
        worker_instance_id=WorkerInstanceId(payload["worker_instance_id"]),
        worker_definition_id=WorkerDefinitionId(payload["worker_definition_id"]),
        definition_revision=DefinitionRevision(payload["definition_revision"]),
        lifecycle_state=WorkerLifecycleState(payload["lifecycle_state"]),
        principal_binding_ref=PrincipalBindingRef(payload["principal_binding_ref"]),
        workspace_context_ref=WorkspaceContextRef(payload["workspace_context_ref"]),
        active_responsibility_refs=tuple(
            ResponsibilityId(value) for value in payload["active_responsibility_refs"]
        ),
        active_goal_refs=tuple(
            WorkerGoalId(value) for value in payload["active_goal_refs"]
        ),
        created_at=_decode_datetime(payload["created_at"]),
        updated_at=_decode_datetime(payload["updated_at"]),
        revision=Revision(payload["revision"]),
    )


def worker_instance_to_json(instance: WorkerInstance) -> str:
    return stable_record_json(worker_instance_to_payload(instance))


def worker_instance_from_json(payload: str) -> WorkerInstance:
    return worker_instance_from_payload(json.loads(payload))


def responsibility_to_payload(responsibility: Responsibility) -> dict[str, Any]:
    return {
        "codec_version": CODEC_VERSION,
        "responsibility_id": responsibility.responsibility_id,
        "worker_instance_id": responsibility.worker_instance_id,
        "objective": responsibility.objective,
        "scope_ref": responsibility.scope_ref,
        "status": responsibility.status.value,
        "assigned_at": _encode_datetime(responsibility.assigned_at),
        "revision": responsibility.revision.value,
    }


def responsibility_from_payload(payload: dict[str, Any]) -> Responsibility:
    if payload.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported Responsibility codec version")
    return Responsibility(
        responsibility_id=ResponsibilityId(payload["responsibility_id"]),
        worker_instance_id=WorkerInstanceId(payload["worker_instance_id"]),
        objective=payload["objective"],
        scope_ref=ResponsibilityScopeRef(payload["scope_ref"]),
        status=ResponsibilityStatus(payload["status"]),
        assigned_at=_decode_datetime(payload["assigned_at"]),
        revision=Revision(payload["revision"]),
    )


def responsibility_to_json(responsibility: Responsibility) -> str:
    return stable_record_json(responsibility_to_payload(responsibility))


def responsibility_from_json(payload: str) -> Responsibility:
    return responsibility_from_payload(json.loads(payload))


def worker_goal_to_payload(goal: WorkerGoal) -> dict[str, Any]:
    return {
        "codec_version": CODEC_VERSION,
        "goal_id": goal.goal_id,
        "responsibility_id": goal.responsibility_id,
        "objective": goal.objective,
        "success_criteria": goal.success_criteria,
        "metric_refs": _encode_string_tuple(goal.metric_refs),
        "sla_slo_refs": _encode_string_tuple(goal.sla_slo_refs),
        "deadline_or_cadence": goal.deadline_or_cadence,
        "priority": goal.priority,
        "status": goal.status.value,
        "progress_projection_ref": goal.progress_projection_ref,
        "evaluation_cadence_ref": goal.evaluation_cadence_ref,
        "revision": goal.revision.value,
    }


def worker_goal_from_payload(payload: dict[str, Any]) -> WorkerGoal:
    if payload.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported WorkerGoal codec version")
    return WorkerGoal(
        goal_id=WorkerGoalId(payload["goal_id"]),
        responsibility_id=ResponsibilityId(payload["responsibility_id"]),
        objective=payload["objective"],
        success_criteria=SuccessCriteriaRef(payload["success_criteria"]),
        metric_refs=tuple(MetricRef(value) for value in payload["metric_refs"]),
        sla_slo_refs=tuple(SlaSloRef(value) for value in payload["sla_slo_refs"]),
        deadline_or_cadence=DeadlineOrCadenceRef(payload["deadline_or_cadence"]),
        priority=payload["priority"],
        status=WorkerGoalStatus(payload["status"]),
        progress_projection_ref=ProgressProjectionRef(payload["progress_projection_ref"]),
        evaluation_cadence_ref=EvaluationCadenceRef(payload["evaluation_cadence_ref"]),
        revision=Revision(payload["revision"]),
    )


def worker_goal_to_json(goal: WorkerGoal) -> str:
    return stable_record_json(worker_goal_to_payload(goal))


def worker_goal_from_json(payload: str) -> WorkerGoal:
    return worker_goal_from_payload(json.loads(payload))


def _encode_progress_checkpoint(
    checkpoint: ProgressCheckpoint | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {
        "checkpointed_at": _encode_datetime(checkpoint.checkpointed_at),
        "summary_ref": checkpoint.summary_ref,
    }


def _decode_progress_checkpoint(payload: dict[str, Any] | None) -> ProgressCheckpoint | None:
    if payload is None:
        return None
    summary_ref = payload.get("summary_ref")
    return ProgressCheckpoint(
        checkpointed_at=_decode_datetime(payload["checkpointed_at"]),
        summary_ref=(
            ProgressCheckpointRef(summary_ref) if summary_ref is not None else None
        ),
    )


def work_continuity_state_to_payload(state: WorkContinuityState) -> dict[str, Any]:
    return {
        "codec_version": CODEC_VERSION,
        "worker_instance_ref": state.worker_instance_ref,
        "responsibility_refs": _encode_string_tuple(state.responsibility_refs),
        "active_goal_refs": _encode_string_tuple(state.active_goal_refs),
        "open_work_refs": _encode_string_tuple(state.open_work_refs),
        "blocked_work_refs": _encode_string_tuple(state.blocked_work_refs),
        "pending_external_refs": _encode_string_tuple(state.pending_external_refs),
        "pending_human_refs": _encode_string_tuple(state.pending_human_refs),
        "recent_decision_refs": _encode_string_tuple(state.recent_decision_refs),
        "relevant_artifact_refs": _encode_string_tuple(state.relevant_artifact_refs),
        "unresolved_problem_refs": _encode_string_tuple(state.unresolved_problem_refs),
        "last_progress_checkpoint": _encode_progress_checkpoint(
            state.last_progress_checkpoint
        ),
        "next_action_hint": state.next_action_hint,
        "context_anchor_refs": _encode_string_tuple(state.context_anchor_refs),
        "revision": state.revision.value,
    }


def work_continuity_state_from_payload(payload: dict[str, Any]) -> WorkContinuityState:
    if payload.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported WorkContinuityState codec version")
    return WorkContinuityState(
        worker_instance_ref=WorkerInstanceId(payload["worker_instance_ref"]),
        responsibility_refs=tuple(
            ResponsibilityId(value) for value in payload["responsibility_refs"]
        ),
        active_goal_refs=tuple(
            WorkerGoalId(value) for value in payload["active_goal_refs"]
        ),
        open_work_refs=tuple(WorkReference(value) for value in payload["open_work_refs"]),
        blocked_work_refs=tuple(
            WorkReference(value) for value in payload["blocked_work_refs"]
        ),
        pending_external_refs=tuple(
            ExternalDependencyReference(value)
            for value in payload["pending_external_refs"]
        ),
        pending_human_refs=tuple(
            HumanPendingReference(value) for value in payload["pending_human_refs"]
        ),
        recent_decision_refs=tuple(
            DecisionId(value) for value in payload["recent_decision_refs"]
        ),
        relevant_artifact_refs=tuple(
            ArtifactReference(value) for value in payload["relevant_artifact_refs"]
        ),
        unresolved_problem_refs=tuple(
            ProblemReference(value) for value in payload["unresolved_problem_refs"]
        ),
        last_progress_checkpoint=_decode_progress_checkpoint(
            payload["last_progress_checkpoint"]
        ),
        next_action_hint=payload["next_action_hint"],
        context_anchor_refs=tuple(
            ContextAnchorReference(value) for value in payload["context_anchor_refs"]
        ),
        revision=Revision(payload["revision"]),
    )


def work_continuity_state_to_json(state: WorkContinuityState) -> str:
    return stable_record_json(work_continuity_state_to_payload(state))


def work_continuity_state_from_json(payload: str) -> WorkContinuityState:
    return work_continuity_state_from_payload(json.loads(payload))


def worker_principal_binding_to_payload(binding: WorkerPrincipalBinding) -> dict[str, Any]:
    return {
        "codec_version": CODEC_VERSION,
        "worker_instance_id": binding.worker_instance_id,
        "tenant_id": binding.tenant_id,
        "workspace_id": binding.workspace_id,
        "principal_id": binding.principal_id,
        "created_at": _encode_datetime(binding.created_at),
        "revision": binding.revision.value,
    }


def worker_principal_binding_from_payload(payload: dict[str, Any]) -> WorkerPrincipalBinding:
    if payload.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported WorkerPrincipalBinding codec version")
    missing = [
        field
        for field in ("tenant_id", "workspace_id", "principal_id")
        if field not in payload
    ]
    if missing:
        raise ValueError(
            "malformed WorkerPrincipalBinding payload: missing scoped identity fields "
            f"{', '.join(missing)}"
        )
    return WorkerPrincipalBinding(
        worker_instance_id=WorkerInstanceId(payload["worker_instance_id"]),
        tenant_id=payload["tenant_id"],
        workspace_id=payload["workspace_id"],
        principal_id=payload["principal_id"],
        created_at=_decode_datetime(payload["created_at"]),
        revision=Revision(payload["revision"]),
    )


def worker_principal_binding_to_json(binding: WorkerPrincipalBinding) -> str:
    return stable_record_json(worker_principal_binding_to_payload(binding))


def worker_principal_binding_from_json(payload: str) -> WorkerPrincipalBinding:
    return worker_principal_binding_from_payload(json.loads(payload))
