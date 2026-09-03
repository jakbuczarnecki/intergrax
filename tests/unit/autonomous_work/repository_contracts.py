# © Artur Czarnecki. All rights reserved.

"""Shared Autonomous Work repository contract suite for in-memory and durable adapters."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

import pytest

from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityConflict,
    AutonomousWorkEntityNotFound,
    AutonomousWorkRevisionConflict,
    ResponsibilityRepository,
    WorkContinuityStateRepository,
    WorkerDefinitionRepository,
    WorkerGoalRepository,
    WorkerInstanceRepository,
)
from intergrax.contracts.autonomous_work import (
    DefinitionRevision,
    ProgressCheckpoint,
    ResponsibilityStatus,
    Revision,
    WorkContinuityState,
    WorkerDefinition,
    WorkerGoal,
    WorkerGoalStatus,
    WorkerInstance,
    WorkerLifecycleState,
    Responsibility,
    initial_definition_revision,
    initial_revision,
    mint_responsibility_id,
    mint_worker_definition_id,
    mint_worker_goal_id,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    CapabilityProfileRef,
    CodecraftProfileRef,
    CollaborationProfileRef,
    EscalationPolicyRef,
    GovernanceProfileRef,
    MemoryProfileRef,
    ObservabilityProfileRef,
    RiskProfileRef,
    ScheduleProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import (
    DeadlineOrCadenceRef,
    DefaultGoalPolicyRef,
    EvaluationCadenceRef,
    MetricRef,
    PrincipalBindingPolicyRef,
    PrincipalBindingRef,
    ProgressProjectionRef,
    ResponsibilityScopeRef,
    ResponsibilityTemplateRef,
    SlaSloRef,
    SuccessCriteriaRef,
    WorkReference,
    WorkspaceContextRef,
    WorkspaceScopeRef,
)
from intergrax.contracts.decision_identity import mint_decision_id

_UTC = timezone.utc


def profile_refs() -> dict[str, object]:
    version = initial_profile_version()
    return {
        "default_goal_policy_ref": DefaultGoalPolicyRef("goal-policy/default"),
        "principal_binding_policy_ref": PrincipalBindingPolicyRef(
            "binding-policy/default"
        ),
        "workspace_scope_ref": WorkspaceScopeRef("workspace-scope/default"),
        "governance_profile_ref": GovernanceProfileRef(
            profile_id="governance/default",
            version=version,
        ),
        "budget_profile_ref": BudgetProfileRef(
            profile_id="budget/default",
            version=version,
        ),
        "memory_profile_ref": MemoryProfileRef(
            profile_id="memory/default",
            version=version,
        ),
        "capability_profile_ref": CapabilityProfileRef(
            profile_id="capability/default",
            version=version,
        ),
        "codecraft_profile_ref": CodecraftProfileRef(
            profile_id="codecraft/default",
            version=version,
        ),
        "risk_profile_ref": RiskProfileRef(
            profile_id="risk/default",
            version=version,
        ),
        "schedule_profile_ref": ScheduleProfileRef(
            profile_id="schedule/default",
            version=version,
        ),
        "escalation_policy_ref": EscalationPolicyRef(
            profile_id="escalation/default",
            version=version,
        ),
        "collaboration_profile_ref": CollaborationProfileRef(
            profile_id="collaboration/default",
            version=version,
        ),
        "observability_profile_ref": ObservabilityProfileRef(
            profile_id="observability/default",
            version=version,
        ),
    }


def worker_definition(**overrides: object) -> WorkerDefinition:
    payload = {
        "worker_definition_id": mint_worker_definition_id(),
        "display_name": "Order Operations Worker",
        "role": "Order Operations Worker",
        "revision": initial_definition_revision(),
        "responsibility_template_refs": (
            ResponsibilityTemplateRef("template/order-ops"),
        ),
        **profile_refs(),
    }
    payload.update(overrides)
    return WorkerDefinition(**payload)


def worker_instance(**overrides: object) -> WorkerInstance:
    now = datetime(2026, 9, 2, 12, 0, tzinfo=_UTC)
    payload = {
        "worker_instance_id": mint_worker_instance_id(),
        "worker_definition_id": mint_worker_definition_id(),
        "definition_revision": DefinitionRevision(1),
        "lifecycle_state": WorkerLifecycleState.PROVISIONING,
        "principal_binding_ref": PrincipalBindingRef("binding/order-ops-1"),
        "workspace_context_ref": WorkspaceContextRef("workspace/order-ops"),
        "active_responsibility_refs": (),
        "active_goal_refs": (),
        "created_at": now,
        "updated_at": now,
        "revision": initial_revision(),
    }
    payload.update(overrides)
    return WorkerInstance(**payload)


def responsibility(**overrides: object) -> Responsibility:
    payload = {
        "responsibility_id": mint_responsibility_id(),
        "worker_instance_id": mint_worker_instance_id(),
        "objective": "Process incoming customer orders according to company policy.",
        "scope_ref": ResponsibilityScopeRef("scope/order-processing"),
        "status": ResponsibilityStatus.ACTIVE,
        "assigned_at": datetime(2026, 9, 2, 12, 0, tzinfo=_UTC),
        "revision": initial_revision(),
    }
    payload.update(overrides)
    return Responsibility(**payload)


def worker_goal(**overrides: object) -> WorkerGoal:
    payload = {
        "goal_id": mint_worker_goal_id(),
        "responsibility_id": mint_responsibility_id(),
        "objective": "Complete 99% of orders within 30 minutes.",
        "success_criteria": SuccessCriteriaRef("criteria/sla-30m"),
        "metric_refs": (MetricRef("metric/order-completion-time"),),
        "sla_slo_refs": (SlaSloRef("slo/order-30m"),),
        "deadline_or_cadence": DeadlineOrCadenceRef("cadence/every-5m"),
        "priority": 10,
        "status": WorkerGoalStatus.ACTIVE,
        "progress_projection_ref": ProgressProjectionRef("projection/sla-30m"),
        "evaluation_cadence_ref": EvaluationCadenceRef("cadence/goal-eval-5m"),
        "revision": initial_revision(),
    }
    payload.update(overrides)
    return WorkerGoal(**payload)


def continuity_state(**overrides: object) -> WorkContinuityState:
    worker_id = mint_worker_instance_id()
    payload = {
        "worker_instance_ref": worker_id,
        "responsibility_refs": (),
        "active_goal_refs": (),
        "open_work_refs": (),
        "blocked_work_refs": (),
        "pending_external_refs": (),
        "pending_human_refs": (),
        "recent_decision_refs": (),
        "relevant_artifact_refs": (),
        "unresolved_problem_refs": (),
        "last_progress_checkpoint": None,
        "next_action_hint": None,
        "context_anchor_refs": (),
        "revision": initial_revision(),
    }
    payload.update(overrides)
    return WorkContinuityState(**payload)


def contract_worker_definition_create_get_and_version_history(
    repo: WorkerDefinitionRepository,
) -> None:
    definition_v0 = worker_definition(revision=DefinitionRevision(0))
    definition_v1 = replace(
        definition_v0,
        revision=DefinitionRevision(1),
        display_name="Order Operations Worker v1",
    )

    created_v0 = repo.create(definition_v0)
    created_v1 = repo.create(definition_v1)

    assert created_v0 == definition_v0
    assert created_v1 == definition_v1
    assert (
        repo.get(
            worker_definition_id=definition_v0.worker_definition_id,
            definition_revision=DefinitionRevision(0),
        )
        == definition_v0
    )
    assert (
        repo.get(
            worker_definition_id=definition_v0.worker_definition_id,
            definition_revision=DefinitionRevision(1),
        )
        == definition_v1
    )


def contract_worker_definition_idempotent_identical_create(
    repo: WorkerDefinitionRepository,
) -> None:
    definition = worker_definition()
    first = repo.create(definition)
    second = repo.create(definition)
    assert second == first


def contract_worker_definition_same_revision_different_content_conflicts(
    repo: WorkerDefinitionRepository,
) -> None:
    definition = worker_definition()
    repo.create(definition)
    conflict = replace(definition, display_name="Different Name")
    with pytest.raises(AutonomousWorkEntityConflict):
        repo.create(conflict)


def contract_worker_definition_missing_returns_none(repo: WorkerDefinitionRepository) -> None:
    definition_id = mint_worker_definition_id()
    assert (
        repo.get(
            worker_definition_id=definition_id,
            definition_revision=DefinitionRevision(0),
        )
        is None
    )


def contract_mutable_repository_create_and_get(
    repo: WorkerInstanceRepository | ResponsibilityRepository | WorkerGoalRepository | WorkContinuityStateRepository,
    entity: WorkerInstance | Responsibility | WorkerGoal | WorkContinuityState,
    *,
    load: Callable[[Any], Any],
) -> None:
    created = repo.create(entity)
    loaded = load(repo)
    assert created == entity
    assert loaded == created


def contract_mutable_repository_idempotent_identical_create(
    repo: WorkerInstanceRepository | ResponsibilityRepository | WorkerGoalRepository | WorkContinuityStateRepository,
    entity: WorkerInstance | Responsibility | WorkerGoal | WorkContinuityState,
) -> None:
    first = repo.create(entity)
    second = repo.create(entity)
    assert second == first


def contract_mutable_repository_same_id_different_content_conflicts(
    repo: WorkerInstanceRepository | ResponsibilityRepository | WorkerGoalRepository | WorkContinuityStateRepository,
    entity: WorkerInstance | Responsibility | WorkerGoal | WorkContinuityState,
    mutator: Callable[[Any], Any],
) -> None:
    repo.create(entity)
    conflict_entity = mutator(entity)
    with pytest.raises(AutonomousWorkEntityConflict):
        repo.create(conflict_entity)


def contract_worker_instance_replace_advances_revision_and_does_not_mutate_input(
    repo: WorkerInstanceRepository,
) -> None:
    instance = worker_instance()
    created = repo.create(instance)
    next_instance = replace(
        created,
        lifecycle_state=WorkerLifecycleState.ACTIVE,
        updated_at=datetime(2026, 9, 2, 13, 0, tzinfo=_UTC),
    )
    assert next_instance.revision == created.revision
    updated = repo.replace(next_instance, expected_revision=created.revision)
    assert updated.revision == Revision(created.revision.value + 1)
    assert next_instance.revision == created.revision
    assert created.revision == initial_revision()
    loaded = repo.get(worker_instance_id=instance.worker_instance_id)
    assert loaded == updated


def contract_worker_instance_stale_replace_conflicts_and_preserves_state(
    repo: WorkerInstanceRepository,
) -> None:
    created = repo.create(worker_instance())
    first_update = repo.replace(
        replace(created, lifecycle_state=WorkerLifecycleState.ACTIVE),
        expected_revision=created.revision,
    )
    with pytest.raises(AutonomousWorkRevisionConflict) as exc_info:
        repo.replace(
            replace(first_update, lifecycle_state=WorkerLifecycleState.WORKING),
            expected_revision=created.revision,
        )
    conflict = exc_info.value
    assert conflict.expected_revision == created.revision
    assert conflict.actual_revision == first_update.revision
    loaded = repo.get(worker_instance_id=created.worker_instance_id)
    assert loaded == first_update


def contract_worker_instance_replace_missing_raises_not_found(
    repo: WorkerInstanceRepository,
) -> None:
    instance = worker_instance()
    with pytest.raises(AutonomousWorkEntityNotFound):
        repo.replace(instance, expected_revision=initial_revision())


def contract_mutable_repository_replace_rejects_candidate_revision_mismatch(
    repo: WorkerInstanceRepository | ResponsibilityRepository | WorkerGoalRepository | WorkContinuityStateRepository,
    entity: WorkerInstance | Responsibility | WorkerGoal | WorkContinuityState,
    candidate: Any,
    *,
    load: Callable[[Any], Any],
) -> None:
    created = repo.create(entity)
    with pytest.raises(ValueError, match="replacement candidate revision"):
        repo.replace(candidate, expected_revision=created.revision)
    assert load(repo) == created


def contract_continuity_repository_returns_latest_committed_state(
    repo: WorkContinuityStateRepository,
) -> None:
    worker_id = mint_worker_instance_id()
    state = continuity_state(
        worker_instance_ref=worker_id,
        open_work_refs=(WorkReference("work/open-1"),),
        recent_decision_refs=(mint_decision_id(),),
        last_progress_checkpoint=ProgressCheckpoint(
            checkpointed_at=datetime(2026, 9, 2, 12, 30, tzinfo=_UTC),
        ),
        next_action_hint="Resume order triage",
    )
    created = repo.create(state)
    updated = repo.replace(
        replace(created, next_action_hint="Continue triage"),
        expected_revision=created.revision,
    )
    loaded = repo.get(worker_instance_id=worker_id)
    assert loaded == updated
    assert loaded is not None
    assert loaded.revision == Revision(1)


def contract_continuity_state_worker_isolation(repo: WorkContinuityStateRepository) -> None:
    worker_a = mint_worker_instance_id()
    worker_b = mint_worker_instance_id()
    state_a = continuity_state(
        worker_instance_ref=worker_a,
        open_work_refs=(WorkReference("work/a"),),
    )
    state_b = continuity_state(
        worker_instance_ref=worker_b,
        open_work_refs=(WorkReference("work/b"),),
    )
    repo.create(state_a)
    repo.create(state_b)
    updated_a = repo.replace(
        replace(state_a, next_action_hint="focus A"),
        expected_revision=initial_revision(),
    )
    loaded_b = repo.get(worker_instance_id=worker_b)
    assert loaded_b == state_b
    assert updated_a.worker_instance_ref == worker_a


def contract_continuity_stale_revision_conflict(repo: WorkContinuityStateRepository) -> None:
    created = repo.create(continuity_state())
    first = repo.replace(
        replace(created, next_action_hint="step-1"),
        expected_revision=created.revision,
    )
    with pytest.raises(AutonomousWorkRevisionConflict):
        repo.replace(
            replace(first, next_action_hint="stale"),
            expected_revision=created.revision,
        )
