# © Artur Czarnecki. All rights reserved.

"""AW-2A — Autonomous Work in-memory repository tests."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, Callable
from datetime import datetime, timezone

import pytest

from intergrax.autonomous_work.in_memory_repository import (
    InMemoryResponsibilityRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerDefinitionRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkerInstanceRepository,
)
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

pytestmark = pytest.mark.unit

_UTC = timezone.utc


def _profile_refs() -> dict[str, object]:
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


def _worker_definition(**overrides: object) -> WorkerDefinition:
    payload = {
        "worker_definition_id": mint_worker_definition_id(),
        "display_name": "Order Operations Worker",
        "role": "Order Operations Worker",
        "revision": initial_definition_revision(),
        "responsibility_template_refs": (
            ResponsibilityTemplateRef("template/order-ops"),
        ),
        **_profile_refs(),
    }
    payload.update(overrides)
    return WorkerDefinition(**payload)


def _worker_instance(**overrides: object) -> WorkerInstance:
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


def _responsibility(**overrides: object) -> Responsibility:
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


def _worker_goal(**overrides: object) -> WorkerGoal:
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


def _continuity_state(**overrides: object) -> WorkContinuityState:
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


@pytest.mark.parametrize(
    ("repo_factory", "protocol_type"),
    [
        (InMemoryWorkerDefinitionRepository, WorkerDefinitionRepository),
        (InMemoryWorkerInstanceRepository, WorkerInstanceRepository),
        (InMemoryResponsibilityRepository, ResponsibilityRepository),
        (InMemoryWorkerGoalRepository, WorkerGoalRepository),
        (InMemoryWorkContinuityStateRepository, WorkContinuityStateRepository),
    ],
)
def test_repository_protocol_is_satisfied(
    repo_factory: type[object],
    protocol_type: type[object],
) -> None:
    repo = repo_factory()
    assert isinstance(repo, protocol_type)


def test_worker_definition_create_get_and_version_history() -> None:
    repo = InMemoryWorkerDefinitionRepository()
    definition_v0 = _worker_definition(revision=DefinitionRevision(0))
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


def test_worker_definition_idempotent_identical_create() -> None:
    repo = InMemoryWorkerDefinitionRepository()
    definition = _worker_definition()
    first = repo.create(definition)
    second = repo.create(definition)
    assert second is first or second == first


def test_worker_definition_same_revision_different_content_conflicts() -> None:
    repo = InMemoryWorkerDefinitionRepository()
    definition = _worker_definition()
    repo.create(definition)
    conflict = replace(definition, display_name="Different Name")
    with pytest.raises(AutonomousWorkEntityConflict):
        repo.create(conflict)


def test_worker_definition_missing_returns_none() -> None:
    repo = InMemoryWorkerDefinitionRepository()
    definition_id = mint_worker_definition_id()
    assert (
        repo.get(
            worker_definition_id=definition_id,
            definition_revision=DefinitionRevision(0),
        )
        is None
    )


@pytest.mark.parametrize(
    "repo_factory",
    [
        InMemoryWorkerInstanceRepository,
        InMemoryResponsibilityRepository,
        InMemoryWorkerGoalRepository,
        InMemoryWorkContinuityStateRepository,
    ],
)
def test_mutable_repository_create_and_get(repo_factory: Any) -> None:
    if repo_factory is InMemoryWorkerInstanceRepository:
        entity = _worker_instance()
        repo = repo_factory()
        created = repo.create(entity)
        loaded = repo.get(worker_instance_id=entity.worker_instance_id)
    elif repo_factory is InMemoryResponsibilityRepository:
        entity = _responsibility()
        repo = repo_factory()
        created = repo.create(entity)
        loaded = repo.get(responsibility_id=entity.responsibility_id)
    elif repo_factory is InMemoryWorkerGoalRepository:
        entity = _worker_goal()
        repo = repo_factory()
        created = repo.create(entity)
        loaded = repo.get(goal_id=entity.goal_id)
    else:
        entity = _continuity_state()
        repo = repo_factory()
        created = repo.create(entity)
        loaded = repo.get(worker_instance_id=entity.worker_instance_ref)

    assert created == entity
    assert loaded == created


@pytest.mark.parametrize(
    "repo_factory",
    [
        InMemoryWorkerInstanceRepository,
        InMemoryResponsibilityRepository,
        InMemoryWorkerGoalRepository,
        InMemoryWorkContinuityStateRepository,
    ],
)
def test_mutable_repository_idempotent_identical_create(repo_factory: Any) -> None:
    if repo_factory is InMemoryWorkerInstanceRepository:
        entity = _worker_instance()
        repo = repo_factory()
        first = repo.create(entity)
        second = repo.create(entity)
        assert second == first
        return

    if repo_factory is InMemoryResponsibilityRepository:
        entity = _responsibility()
        repo = repo_factory()
        first = repo.create(entity)
        second = repo.create(entity)
        assert second == first
        return

    if repo_factory is InMemoryWorkerGoalRepository:
        entity = _worker_goal()
        repo = repo_factory()
        first = repo.create(entity)
        second = repo.create(entity)
        assert second == first
        return

    entity = _continuity_state()
    repo = repo_factory()
    first = repo.create(entity)
    second = repo.create(entity)
    assert second == first


@pytest.mark.parametrize(
    ("repo_factory", "mutator"),
    [
        (
            InMemoryWorkerInstanceRepository,
            lambda entity: replace(entity, lifecycle_state=WorkerLifecycleState.ACTIVE),
        ),
        (
            InMemoryResponsibilityRepository,
            lambda entity: replace(entity, objective="Different objective text."),
        ),
        (
            InMemoryWorkerGoalRepository,
            lambda entity: replace(entity, priority=99),
        ),
        (
            InMemoryWorkContinuityStateRepository,
            lambda entity: replace(
                entity,
                open_work_refs=(WorkReference("work/other"),),
            ),
        ),
    ],
)
def test_mutable_repository_same_id_different_content_conflicts(
    repo_factory: Any,
    mutator: Callable[[Any], Any],
) -> None:
    entity = {
        InMemoryWorkerInstanceRepository: _worker_instance,
        InMemoryResponsibilityRepository: _responsibility,
        InMemoryWorkerGoalRepository: _worker_goal,
        InMemoryWorkContinuityStateRepository: _continuity_state,
    }[repo_factory]()
    repo = repo_factory()
    repo.create(entity)
    conflict_entity = mutator(entity)
    with pytest.raises(AutonomousWorkEntityConflict):
        repo.create(conflict_entity)


def test_worker_instance_replace_advances_revision_and_does_not_mutate_input() -> None:
    repo = InMemoryWorkerInstanceRepository()
    instance = _worker_instance()
    created = repo.create(instance)
    next_instance = replace(
        created,
        lifecycle_state=WorkerLifecycleState.ACTIVE,
        updated_at=datetime(2026, 9, 2, 13, 0, tzinfo=_UTC),
    )
    updated = repo.replace(next_instance, expected_revision=created.revision)
    assert updated.revision == Revision(1)
    assert created.revision == initial_revision()
    loaded = repo.get(worker_instance_id=instance.worker_instance_id)
    assert loaded == updated


def test_worker_instance_stale_replace_conflicts_and_preserves_state() -> None:
    repo = InMemoryWorkerInstanceRepository()
    created = repo.create(_worker_instance())
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


def test_worker_instance_replace_missing_raises_not_found() -> None:
    repo = InMemoryWorkerInstanceRepository()
    instance = _worker_instance()
    with pytest.raises(AutonomousWorkEntityNotFound):
        repo.replace(instance, expected_revision=initial_revision())


def test_worker_instance_concurrent_replace_one_wins() -> None:
    repo = InMemoryWorkerInstanceRepository()
    created = repo.create(_worker_instance())
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)
    results: list[WorkerInstance] = []

    def attempt() -> None:
        try:
            barrier.wait(timeout=5)
            result = repo.replace(
                replace(created, lifecycle_state=WorkerLifecycleState.ACTIVE),
                expected_revision=created.revision,
            )
            results.append(result)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], AutonomousWorkRevisionConflict)
    assert len(results) == 1
    assert results[0].revision == Revision(1)
    loaded = repo.get(worker_instance_id=created.worker_instance_id)
    assert loaded == results[0]


def test_continuity_state_restart_safe_latest_committed_state() -> None:
    worker_id = mint_worker_instance_id()
    state = _continuity_state(
        worker_instance_ref=worker_id,
        open_work_refs=(WorkReference("work/open-1"),),
        recent_decision_refs=(mint_decision_id(),),
        last_progress_checkpoint=ProgressCheckpoint(
            checkpointed_at=datetime(2026, 9, 2, 12, 30, tzinfo=_UTC),
        ),
        next_action_hint="Resume order triage",
    )
    repo = InMemoryWorkContinuityStateRepository()
    created = repo.create(state)
    updated = repo.replace(
        replace(created, next_action_hint="Continue triage"),
        expected_revision=created.revision,
    )
    loaded = repo.get(worker_instance_id=worker_id)
    assert loaded == updated
    assert loaded is not None
    assert loaded.revision == Revision(1)


def test_continuity_state_worker_isolation() -> None:
    repo = InMemoryWorkContinuityStateRepository()
    worker_a = mint_worker_instance_id()
    worker_b = mint_worker_instance_id()
    state_a = _continuity_state(
        worker_instance_ref=worker_a,
        open_work_refs=(WorkReference("work/a"),),
    )
    state_b = _continuity_state(
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


def test_continuity_state_survives_unrelated_worker_creation() -> None:
    repo = InMemoryWorkContinuityStateRepository()
    worker_id = mint_worker_instance_id()
    state = _continuity_state(worker_instance_ref=worker_id)
    created = repo.create(state)
    repo.create(_continuity_state())
    loaded = repo.get(worker_instance_id=worker_id)
    assert loaded == created


def test_continuity_stale_revision_conflict() -> None:
    repo = InMemoryWorkContinuityStateRepository()
    created = repo.create(_continuity_state())
    first = repo.replace(
        replace(created, next_action_hint="step-1"),
        expected_revision=created.revision,
    )
    with pytest.raises(AutonomousWorkRevisionConflict):
        repo.replace(
            replace(first, next_action_hint="stale"),
            expected_revision=created.revision,
        )
