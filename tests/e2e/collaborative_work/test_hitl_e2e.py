# © Artur Czarnecki. All rights reserved.

"""HITL / governed continuation E2E through canonical platform mechanisms."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.human.governed_continuation_grant import GovernedContinuationGrantCoordinator
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from tests.e2e.collaborative_work.harness.composition import MultiplayerE2EContext
from tests.e2e.collaborative_work.harness.constants import (
    POLICY_BUNDLE_D1,
    POLICY_BUNDLE_D2,
    POLICY_BUNDLE_ID,
    POLICY_BUNDLE_V1,
    POLICY_BUNDLE_V2,
    POLICY_RULE_HITL,
    PRINCIPAL_ALICE,
    RESOURCE_A,
    SIDE_EFFECT_DIGEST_1,
    SIDE_EFFECT_SCOPE_1,
    SIDE_EFFECT_SCOPE_2,
    TENANT_A,
)
from tests.e2e.collaborative_work.harness.fixtures import seed_direct_allow_fixture
from tests.e2e.collaborative_work.harness.scenario_runner import (
    assert_authorization_allowed,
    assert_require_human_paused,
    build_enforcement_request,
    require_human_runtime_decision,
    run_multiplayer_e2e_scenario,
)
from tests.e2e.collaborative_work.harness.side_effect_probe import SideEffectProbe

pytestmark = pytest.mark.e2e

_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_OTHER_RUN = mint_run_id()
_APPROVER = local_development_approver_evidence(tenant_id=TENANT_A)


def _task() -> Task:
    return Task(
        tenant_id=TENANT_A,
        user_id=PRINCIPAL_ALICE,
        message="e2e-hitl",
        task_id=_TASK_ID,
    )


def _lifecycle(task: Task) -> TaskLifecycle:
    lifecycle = TaskLifecycle()
    lifecycle.transition(task, TaskState.CLASSIFIED)
    lifecycle.transition(task, TaskState.PLANNED)
    return lifecycle


def _configure_hitl(context: MultiplayerE2EContext) -> None:
    context.runtime_policy.set_decision(
        require_human_runtime_decision(
            policy_rule_id=POLICY_RULE_HITL,
            policy_bundle_id=POLICY_BUNDLE_ID,
            policy_bundle_version=POLICY_BUNDLE_V1,
            policy_bundle_digest=POLICY_BUNDLE_D1,
        )
    )


def _approve_paused_task(task: Task, *, run_id: str = _RUN_ID) -> GovernedContinuationApprovalGrant:
    pause = task.runtime.governance.pause_record
    assert pause is not None
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=_APPROVER,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
        run_id=run_id,
    )
    grant = GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert grant is not None
    return grant


def test_sqlite_require_human_pauses_before_side_effect(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    _configure_hitl(sqlite_e2e_context)
    task = _task()
    lifecycle = _lifecycle(task)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(task_id=_TASK_ID, run_id=_RUN_ID),
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    assert_require_human_paused(result, probe, task=task)
    assert task.state is TaskState.WAITING_FOR_HUMAN


def test_sqlite_hitl_approve_grant_execute_once(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    _configure_hitl(sqlite_e2e_context)
    task = _task()
    lifecycle = _lifecycle(task)
    probe = SideEffectProbe()
    request = build_enforcement_request(task_id=_TASK_ID, run_id=_RUN_ID)

    first = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    assert_require_human_paused(first, probe, task=task)
    grant = _approve_paused_task(task)
    assert grant.side_effect_scope_id == SIDE_EFFECT_SCOPE_1
    assert grant.resource_scope == RESOURCE_A
    assert grant.task_id == _TASK_ID
    assert grant.run_id == _RUN_ID

    lifecycle.transition(task, TaskState.RUNNING)
    second = run_multiplayer_e2e_scenario(sqlite_e2e_context, request, probe, task=task)
    assert_authorization_allowed(second, probe)
    assert task.runtime.governance.governed_continuation_grant is None

    lifecycle.transition(task, TaskState.RUNNING)
    third = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    assert isinstance(third, MeaningfulSideEffectAuthorizationResult)
    assert third.decision.action is PolicyAction.REQUIRE_HUMAN
    assert probe.count == 1


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_hitl_approve_grant_execute_once(
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(postgresql_e2e_context.bundle)
    _configure_hitl(postgresql_e2e_context)
    task = _task()
    lifecycle = _lifecycle(task)
    probe = SideEffectProbe()
    request = build_enforcement_request(task_id=_TASK_ID, run_id=_RUN_ID)
    run_multiplayer_e2e_scenario(
        postgresql_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    _approve_paused_task(task)
    lifecycle.transition(task, TaskState.RUNNING)
    result = run_multiplayer_e2e_scenario(postgresql_e2e_context, request, probe, task=task)
    assert_authorization_allowed(result, probe)


def test_sqlite_grant_replay_wrong_run_blocked(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    _configure_hitl(sqlite_e2e_context)
    task = _task()
    lifecycle = _lifecycle(task)
    probe = SideEffectProbe()
    request = build_enforcement_request(task_id=_TASK_ID, run_id=_RUN_ID)
    run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    grant = _approve_paused_task(task)
    task.runtime.governance.governed_continuation_grant = grant.model_copy(
        update={"run_id": _OTHER_RUN},
    )
    lifecycle.transition(task, TaskState.RUNNING)
    result = run_multiplayer_e2e_scenario(sqlite_e2e_context, request, probe, task=task)
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert probe.count == 0


def test_sqlite_grant_replay_wrong_side_effect_scope_blocked(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    _configure_hitl(sqlite_e2e_context)
    task = _task()
    lifecycle = _lifecycle(task)
    probe = SideEffectProbe()
    request = build_enforcement_request(task_id=_TASK_ID, run_id=_RUN_ID)
    run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    _approve_paused_task(task)
    lifecycle.transition(task, TaskState.RUNNING)
    replay_request = build_enforcement_request(
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        side_effect_scope_id=SIDE_EFFECT_SCOPE_2,
        side_effect_scope_digest=SIDE_EFFECT_DIGEST_1,
    )
    result = run_multiplayer_e2e_scenario(sqlite_e2e_context, replay_request, probe, task=task)
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert probe.count == 0


def test_sqlite_stale_policy_bundle_invalidates_grant(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    _configure_hitl(sqlite_e2e_context)
    task = _task()
    lifecycle = _lifecycle(task)
    probe = SideEffectProbe()
    request = build_enforcement_request(task_id=_TASK_ID, run_id=_RUN_ID)
    run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    _approve_paused_task(task)
    sqlite_e2e_context.runtime_policy.set_decision(
        require_human_runtime_decision(
            policy_rule_id=POLICY_RULE_HITL,
            policy_bundle_id=POLICY_BUNDLE_ID,
            policy_bundle_version=POLICY_BUNDLE_V2,
            policy_bundle_digest=POLICY_BUNDLE_D2,
        )
    )
    lifecycle.transition(task, TaskState.RUNNING)
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.REQUIRE_HUMAN
    assert probe.count == 0
    assert task.runtime.governance.governed_continuation_grant is None


def test_sqlite_side_effect_scope_id_distinct_from_resource_scope(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    """ENFORCE-SCOPE-1 — downstream matching treats dimensions separately."""
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    _configure_hitl(sqlite_e2e_context)
    task = _task()
    lifecycle = _lifecycle(task)
    probe = SideEffectProbe()
    request = build_enforcement_request(
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        resource_scope=RESOURCE_A,
        side_effect_scope_id=SIDE_EFFECT_SCOPE_1,
        side_effect_resource=RESOURCE_A,
    )
    run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        request,
        probe,
        task=task,
        lifecycle=lifecycle,
    )
    grant = _approve_paused_task(task)
    assert grant.resource_scope == RESOURCE_A
    assert grant.side_effect_scope_id == SIDE_EFFECT_SCOPE_1
    assert grant.side_effect_scope_id != grant.resource_scope

    lifecycle.transition(task, TaskState.RUNNING)
    mismatched_request = build_enforcement_request(
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        resource_scope=RESOURCE_A,
        side_effect_scope_id=SIDE_EFFECT_SCOPE_2,
        side_effect_scope_digest=SIDE_EFFECT_DIGEST_1,
    )
    result = run_multiplayer_e2e_scenario(sqlite_e2e_context, mismatched_request, probe, task=task)
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert probe.count == 0
