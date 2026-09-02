# © Artur Czarnecki. All rights reserved.

"""AW-1A — Autonomous Work core semantic contract tests."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

import pytest

from intergrax.contracts.autonomous_work import (
    CANONICAL_WORKER_LIFECYCLE_STATES,
    DefinitionRevision,
    ProgressCheckpoint,
    Responsibility,
    ResponsibilityStatus,
    Revision,
    WorkContinuityState,
    WorkerDefinition,
    WorkerGoal,
    WorkerGoalStatus,
    WorkerInstance,
    WorkerLifecycleState,
    initial_definition_revision,
    initial_revision,
    mint_responsibility_id,
    mint_worker_definition_id,
    mint_worker_goal_id,
    mint_worker_instance_id,
    validate_responsibility_id,
    validate_worker_definition_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import (
    ArtifactReference,
    BudgetProfileRef,
    ContextAnchorReference,
    ExternalDependencyReference,
    HumanPendingReference,
    ProblemReference,
    WorkReference,
    CapabilityProfileRef,
    CodecraftProfileRef,
    CollaborationProfileRef,
    DeadlineOrCadenceRef,
    DefaultGoalPolicyRef,
    EscalationPolicyRef,
    EvaluationCadenceRef,
    GovernanceProfileRef,
    MemoryProfileRef,
    MetricRef,
    ObservabilityProfileRef,
    PrincipalBindingPolicyRef,
    PrincipalBindingRef,
    ProgressProjectionRef,
    ResponsibilityScopeRef,
    ResponsibilityTemplateRef,
    RiskProfileRef,
    ScheduleProfileRef,
    SlaSloRef,
    SuccessCriteriaRef,
    WorkspaceContextRef,
    WorkspaceScopeRef,
)
from intergrax.contracts.decision_identity import mint_decision_id

_UTC = timezone.utc


def _profile_refs() -> dict[str, object]:
    return {
        "default_goal_policy_ref": DefaultGoalPolicyRef("goal-policy/default"),
        "principal_binding_policy_ref": PrincipalBindingPolicyRef(
            "binding-policy/default"
        ),
        "workspace_scope_ref": WorkspaceScopeRef("workspace-scope/default"),
        "governance_profile_ref": GovernanceProfileRef("governance/default"),
        "budget_profile_ref": BudgetProfileRef("budget/default"),
        "memory_profile_ref": MemoryProfileRef("memory/default"),
        "capability_profile_ref": CapabilityProfileRef("capability/default"),
        "codecraft_profile_ref": CodecraftProfileRef("codecraft/default"),
        "risk_profile_ref": RiskProfileRef("risk/default"),
        "schedule_profile_ref": ScheduleProfileRef("schedule/default"),
        "escalation_policy_ref": EscalationPolicyRef("escalation/default"),
        "collaboration_profile_ref": CollaborationProfileRef("collaboration/default"),
        "observability_profile_ref": ObservabilityProfileRef("observability/default"),
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


@pytest.mark.unit
def test_ids_accept_valid_and_reject_empty() -> None:
    definition_id = mint_worker_definition_id()
    instance_id = mint_worker_instance_id()
    responsibility_id = mint_responsibility_id()
    goal_id = mint_worker_goal_id()

    assert validate_worker_definition_id(definition_id) == definition_id
    assert validate_worker_instance_id(instance_id) == instance_id
    assert validate_responsibility_id(responsibility_id) == responsibility_id
    assert validate_worker_goal_id(goal_id) == goal_id

    with pytest.raises(ValueError):
        validate_worker_definition_id("")
    with pytest.raises(ValueError):
        validate_worker_instance_id("   ")
    with pytest.raises(ValueError):
        validate_responsibility_id("bad-prefix_0123456789abcdef0123456789abcdef")
    with pytest.raises(TypeError):
        validate_worker_goal_id(123)


@pytest.mark.unit
def test_id_types_are_not_semantically_interchangeable() -> None:
    goal_id = mint_worker_goal_id()
    with pytest.raises(ValueError, match="WorkerInstanceId"):
        validate_worker_instance_id(goal_id)


@pytest.mark.unit
def test_revision_validates_non_negative_and_rejects_negative() -> None:
    revision = Revision(0)
    assert revision.value == 0
    assert Revision(3).value == 3
    with pytest.raises(ValueError, match="non-negative"):
        Revision(-1)
    with pytest.raises(TypeError):
        Revision(True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_revision_is_immutable_and_comparable() -> None:
    left = Revision(1)
    right = Revision(2)
    assert left < right
    with pytest.raises(AttributeError):
        left.value = 3  # type: ignore[misc]


@pytest.mark.unit
def test_lifecycle_states_match_canonical_list() -> None:
    assert (
        tuple(state.value for state in WorkerLifecycleState)
        == CANONICAL_WORKER_LIFECYCLE_STATES
    )
    assert len(WorkerLifecycleState) == 11


@pytest.mark.unit
def test_worker_definition_minimal_valid_construction() -> None:
    definition = _worker_definition()
    assert definition.display_name == "Order Operations Worker"
    assert definition.role == "Order Operations Worker"
    assert definition.responsibility_template_refs == (
        ResponsibilityTemplateRef("template/order-ops"),
    )


@pytest.mark.unit
def test_worker_definition_rejects_invalid_required_fields() -> None:
    with pytest.raises(ValueError, match="display_name"):
        _worker_definition(display_name="  ")
    with pytest.raises(ValueError, match="role"):
        _worker_definition(role="")
    with pytest.raises(ValueError, match="GovernanceProfileRef"):
        _worker_definition(governance_profile_ref=GovernanceProfileRef("  "))


@pytest.mark.unit
def test_worker_definition_role_is_descriptive_only() -> None:
    _worker_definition()
    field_names = {field.name for field in fields(WorkerDefinition)}
    assert "role" in field_names
    assert "authority" not in field_names
    assert "permissions" not in field_names
    assert "credentials" not in field_names


@pytest.mark.unit
def test_worker_instance_links_definition_and_lifecycle() -> None:
    definition_id = mint_worker_definition_id()
    responsibility_id = mint_responsibility_id()
    goal_id = mint_worker_goal_id()
    instance = _worker_instance(
        worker_definition_id=definition_id,
        lifecycle_state=WorkerLifecycleState.ACTIVE,
        active_responsibility_refs=(responsibility_id,),
        active_goal_refs=(goal_id,),
    )
    assert instance.worker_definition_id == definition_id
    assert instance.lifecycle_state is WorkerLifecycleState.ACTIVE
    assert instance.active_responsibility_refs == (responsibility_id,)
    assert instance.active_goal_refs == (goal_id,)


@pytest.mark.unit
def test_worker_instance_rejects_naive_datetime() -> None:
    naive = datetime(2026, 9, 2, 12, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        _worker_instance(created_at=naive, updated_at=naive)


@pytest.mark.unit
def test_worker_instance_refs_are_immutable() -> None:
    instance = _worker_instance(active_responsibility_refs=[mint_responsibility_id()])
    assert isinstance(instance.active_responsibility_refs, tuple)
    with pytest.raises(AttributeError):
        instance.lifecycle_state = WorkerLifecycleState.WORKING  # type: ignore[misc]


@pytest.mark.unit
def test_responsibility_valid_contract_and_required_objective() -> None:
    worker_id = mint_worker_instance_id()
    responsibility = _responsibility(worker_instance_id=worker_id)
    assert responsibility.worker_instance_id == worker_id
    assert responsibility.objective
    with pytest.raises(ValueError, match="objective"):
        _responsibility(objective="  ")


@pytest.mark.unit
def test_responsibility_has_no_authority_semantics() -> None:
    _responsibility()
    field_names = {field.name for field in fields(Responsibility)}
    assert "permissions" not in field_names
    assert "authority" not in field_names
    assert "credentials" not in field_names


@pytest.mark.unit
def test_worker_goal_valid_contract_and_responsibility_link() -> None:
    responsibility_id = mint_responsibility_id()
    goal = _worker_goal(responsibility_id=responsibility_id)
    assert goal.responsibility_id == responsibility_id
    assert goal.success_criteria == SuccessCriteriaRef("criteria/sla-30m")


@pytest.mark.unit
def test_worker_goal_is_not_prompt_or_task() -> None:
    _worker_goal()
    field_names = {field.name for field in fields(WorkerGoal)}
    assert "prompt" not in field_names
    assert "task_id" not in field_names
    assert "permissions" not in field_names


@pytest.mark.unit
def test_work_continuity_state_empty_orientation() -> None:
    worker_id = mint_worker_instance_id()
    continuity = WorkContinuityState(
        worker_instance_ref=worker_id,
        responsibility_refs=(),
        active_goal_refs=(),
        open_work_refs=(),
        blocked_work_refs=(),
        pending_external_refs=(),
        pending_human_refs=(),
        recent_decision_refs=(),
        relevant_artifact_refs=(),
        unresolved_problem_refs=(),
        last_progress_checkpoint=None,
        next_action_hint=None,
        context_anchor_refs=(),
        revision=initial_revision(),
    )
    assert continuity.worker_instance_ref == worker_id
    assert continuity.last_progress_checkpoint is None


@pytest.mark.unit
def test_work_continuity_state_populated_orientation() -> None:
    worker_id = mint_worker_instance_id()
    responsibility_id = mint_responsibility_id()
    goal_id = mint_worker_goal_id()
    checkpoint = ProgressCheckpoint(
        checkpointed_at=datetime(2026, 9, 2, 12, 30, tzinfo=_UTC),
    )
    continuity = WorkContinuityState(
        worker_instance_ref=worker_id,
        responsibility_refs=(responsibility_id,),
        active_goal_refs=(goal_id,),
        open_work_refs=(WorkReference("work/open-1"),),
        blocked_work_refs=(WorkReference("work/blocked-1"),),
        pending_external_refs=(ExternalDependencyReference("external/vendor-api"),),
        pending_human_refs=(HumanPendingReference("human/approval-1"),),
        recent_decision_refs=(mint_decision_id(),),
        relevant_artifact_refs=(ArtifactReference("artifact/report-1"),),
        unresolved_problem_refs=(ProblemReference("problem/integration-timeout"),),
        last_progress_checkpoint=checkpoint,
        next_action_hint="Resume supplier synchronization for order batch 42.",
        context_anchor_refs=(ContextAnchorReference("anchor/order-batch-42"),),
        revision=Revision(2),
    )
    assert continuity.revision.value == 2
    assert continuity.next_action_hint is not None


@pytest.mark.unit
def test_work_continuity_state_has_no_history_or_trace_fields() -> None:
    field_names = {field.name for field in fields(WorkContinuityState)}
    forbidden = {
        "conversation_history",
        "chat_history",
        "execution_trace",
        "prompt_history",
        "memory_store",
        "rag_index",
        "full_history",
    }
    assert forbidden.isdisjoint(field_names)


@pytest.mark.unit
def test_public_contracts_avoid_dict_and_any() -> None:
    public_types = (
        WorkerDefinition,
        WorkerInstance,
        Responsibility,
        WorkerGoal,
        WorkContinuityState,
    )
    for contract_type in public_types:
        for field in fields(contract_type):
            hint_name = getattr(field.type, "__name__", str(field.type))
            assert hint_name != "Any"
            assert "dict[" not in str(field.type)
