# © Artur Czarnecki. All rights reserved.

"""G5C-2B-1 — exact side-effect scope identity + scoped approval grant."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import (
    QuoteAcceptanceEvidence,
    quote_acceptance_side_effect_scope_digest,
)
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
)
from intergrax.contracts.governed_continuation_correlation import GovernedContinuationCorrelation
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.human.governed_continuation_bridge import (
    bridge_governed_continuation_to_execution_result,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
    GovernedContinuationGrantError,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanApprovalResolutionError, HumanPauseCoordinator
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import HumanApprovalResolution, TaskPauseRecord
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from tests.unit.runtime.human.test_g5b_hitl_resolution import (
    _build_intake_runner_with_hitl,
    _patch_hitl_runtime_events,
    _set_human_response,
    bound_hitl_test_execution_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TASK_ID = mint_task_id()
RUN_ID = mint_run_id()
RUN_OTHER = mint_run_id()
TASK_OTHER = mint_task_id()
ATTEMPT_ID = mint_attempt_id()
EXECUTION_ID = mint_execution_id()
OPERATION = "collaborative.document.delete"
RESOURCE = "document-123"
POLICY_RULE = "runtime.hitl"
BUNDLE_ID = "bundle-grant-test"
BUNDLE_VERSION = "1.0.0"
BUNDLE_DIGEST = "sha256:" + ("11" * 32)
SCOPE_1 = "side-effect-scope-1"
SCOPE_2 = "side-effect-scope-2"
SCOPE_DIGEST_1 = "sha256:" + ("ab" * 32)
SCOPE_DIGEST_2 = "sha256:" + ("cd" * 32)
QUOTE_SCOPE_DIGEST_1 = quote_acceptance_side_effect_scope_digest(
    QuoteAcceptanceEvidence(
        acceptance_id="acc-1",
        quote_id="quote-grant-1",
        quote_version=1,
        scope_digest=SCOPE_DIGEST_1,
        actor=ActorIdentity(
            kind=ActorKind.USER,
            actor_id="principal-grant-1",
            tenant_id="tenant-a",
        ),
        accepted_at="2026-08-18T12:00:00+00:00",
    )
)
CONTINUATION_1 = "gcr_scope_1"
CONTINUATION_2 = "gcr_scope_2"
PAUSE_A = "pause-a"
PAUSE_B = "pause-b"
HR_A = "hr-a"
HR_B = "hr-b"
APPROVER = local_development_approver_evidence(tenant_id="t1")


def _continuation_request(
    *,
    continuation_request_id: str,
    side_effect_scope_id: str,
    operation_id: str = OPERATION,
    resource_scope: str = RESOURCE,
    task_id: str = TASK_ID,
    run_id: str = RUN_ID,
    side_effect_scope_digest: str | None = None,
    policy_bundle_id: str = BUNDLE_ID,
    policy_bundle_version: str = BUNDLE_VERSION,
    policy_bundle_digest: str = BUNDLE_DIGEST,
) -> GovernedContinuationRequest:
    return GovernedContinuationRequest(
        reason=ContinuationReason.COMPLIANCE,
        task_id=task_id,
        run_id=run_id,
        source_agent_id="agent-test",
        prompt="continuation required",
        continuation_request_id=continuation_request_id,
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        operation_id=operation_id,
        policy_rule_id=POLICY_RULE,
        policy_bundle_id=policy_bundle_id,
        policy_bundle_version=policy_bundle_version,
        policy_bundle_digest=policy_bundle_digest,
        resource_scope=resource_scope,
        policy_action=PolicyAction.REQUIRE_HUMAN,
        correlation={
            "side_effect_scope_id": "evil",
            "side_effect_scope_digest": "evil_digest",
            "run_id": "evil_run",
            "operation_id": "evil",
        },
        context={
            "side_effect_scope_id": "evil",
            "side_effect_scope_digest": "evil_digest",
            "run_id": "evil_run",
            "operation_id": "evil",
        },
    )


def _task() -> Task:
    return Task(tenant_id="t1", user_id="u1", message="x", task_id=TASK_ID)


def _apply_governed_pause(
    task: Task,
    continuation: GovernedContinuationRequest,
) -> TaskPauseRecord:
    execution = bridge_governed_continuation_to_execution_result(continuation)
    HumanPauseCoordinator.apply_pause(task, execution)
    assert task.runtime.governance.pause_record is not None
    return task.runtime.governance.pause_record


def _approve_resolution(
    task: Task,
    *,
    pause_id: str,
    human_request_id: str,
    run_id: str = RUN_ID,
) -> HumanApprovalResolution:
    return HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=APPROVER,
        pause_id=pause_id,
        human_request_id=human_request_id,
        run_id=run_id,
    )


def test_same_action_resource_different_scope_ids_remain_distinct() -> None:
    continuation_s1 = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    continuation_s2 = _continuation_request(
        continuation_request_id=CONTINUATION_2,
        side_effect_scope_id=SCOPE_2,
        operation_id=OPERATION,
        resource_scope=RESOURCE,
    )

    task_s1 = _task()
    pause_s1 = _apply_governed_pause(task_s1, continuation_s1)
    _approve_resolution(
        task_s1,
        pause_id=pause_s1.pause_id,
        human_request_id=pause_s1.human_request_id,
    )
    grant_s1 = GovernedContinuationGrantCoordinator.create_grant_from_approval(task_s1)
    assert grant_s1 is not None
    assert grant_s1.side_effect_scope_id == SCOPE_1
    assert grant_s1.side_effect_scope_id != SCOPE_2

    corr_s2 = continuation_s2.to_correlation()
    assert corr_s2.side_effect_scope_id == SCOPE_2
    assert corr_s2.task_id == TASK_ID
    assert corr_s2.run_id == RUN_ID
    assert corr_s2.operation_id == OPERATION
    assert corr_s2.resource_scope == RESOURCE
    assert grant_s1.continuation_request_id != corr_s2.continuation_request_id


def test_run_mismatch_fails_closed() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
        run_id=RUN_ID,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    _approve_resolution(
        task,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
        run_id=RUN_OTHER,
    )
    with pytest.raises(GovernedContinuationGrantError, match="continuation run_id mismatch"):
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert task.runtime.governance.governed_continuation_grant is None


def test_matching_run_creates_grant_with_original_run_id() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
        run_id=RUN_ID,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    _approve_resolution(
        task,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
        run_id=RUN_ID,
    )
    grant = GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert grant is not None
    assert grant.run_id == RUN_ID


def test_task_mismatch_fails_closed() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
        task_id=TASK_OTHER,
        run_id=RUN_ID,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    _approve_resolution(
        task,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
        run_id=RUN_ID,
    )
    with pytest.raises(GovernedContinuationGrantError, match="continuation task_id mismatch"):
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert task.runtime.governance.governed_continuation_grant is None


def test_exact_canonical_approve_creates_grant() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    resolution = _approve_resolution(
        task,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )
    grant = GovernedContinuationGrantCoordinator.create_grant_from_approval(task)

    assert isinstance(grant, GovernedContinuationApprovalGrant)
    assert task.runtime.governance.hitl_resolution == resolution
    assert grant.task_id == TASK_ID
    assert grant.run_id == RUN_ID
    assert grant.pause_id == pause.pause_id
    assert grant.human_request_id == pause.human_request_id
    assert grant.continuation_request_id == CONTINUATION_1
    assert grant.side_effect_scope_id == SCOPE_1
    assert grant.operation_id == OPERATION
    assert grant.resource_scope == RESOURCE
    assert grant.policy_rule_id == POLICY_RULE
    assert grant.policy_bundle_id == BUNDLE_ID
    assert grant.policy_bundle_version == BUNDLE_VERSION
    assert grant.policy_bundle_digest == BUNDLE_DIGEST
    assert task.runtime.governance.governed_continuation_grant == grant


def test_stale_pause_cannot_create_grant() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)

    with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.APPROVE,
            approver=APPROVER,
            pause_id=PAUSE_B,
            human_request_id=pause.human_request_id,
            run_id=RUN_ID,
        )

    assert task.runtime.governance.governed_continuation_grant is None
    assert task.runtime.governance.hitl_resolution is None

    with pytest.raises(HumanApprovalResolutionError, match="human_request_id mismatch"):
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.APPROVE,
            approver=APPROVER,
            pause_id=pause.pause_id,
            human_request_id=HR_B,
            run_id=RUN_ID,
        )

    assert task.runtime.governance.governed_continuation_grant is None
    assert task.runtime.governance.hitl_resolution is None


def test_reject_does_not_create_grant() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.REJECT,
        approver=APPROVER,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )
    with pytest.raises(GovernedContinuationGrantError, match="verdict is not approve"):
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert task.runtime.governance.governed_continuation_grant is None


def test_escalate_does_not_create_grant() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.ESCALATE,
        approver=APPROVER,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )
    with pytest.raises(GovernedContinuationGrantError, match="verdict is not approve"):
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert task.runtime.governance.governed_continuation_grant is None


def test_generic_hitl_approve_does_not_create_grant() -> None:
    task = _task()
    task.runtime.governance.paused = True
    task.runtime.governance.pause_record = TaskPauseRecord(
        pause_id=PAUSE_A,
        task_id=TASK_ID,
        human_request_id=HR_A,
    )
    task.runtime.governance.human_request = HumanRequest(
        request_id=HR_A,
        prompt="generic approval?",
    )
    _approve_resolution(task, pause_id=PAUSE_A, human_request_id=HR_A)
    grant = GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert grant is None
    assert task.runtime.governance.governed_continuation_grant is None


def test_continuation_request_id_alone_is_insufficient() -> None:
    shared_continuation_id = "gcr_shared"
    continuation_s1 = _continuation_request(
        continuation_request_id=shared_continuation_id,
        side_effect_scope_id=SCOPE_1,
    )
    continuation_s2 = _continuation_request(
        continuation_request_id=shared_continuation_id,
        side_effect_scope_id=SCOPE_2,
    )
    corr_s1 = continuation_s1.to_correlation()
    corr_s2 = continuation_s2.to_correlation()
    assert corr_s1.continuation_request_id == corr_s2.continuation_request_id
    assert corr_s1.side_effect_scope_id != corr_s2.side_effect_scope_id


def test_dynamic_context_cannot_change_grant_scope() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
        side_effect_scope_digest=SCOPE_DIGEST_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    _approve_resolution(
        task,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )
    grant = GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert grant is not None
    assert grant.side_effect_scope_id == SCOPE_1
    assert grant.side_effect_scope_digest == SCOPE_DIGEST_1
    assert grant.run_id == RUN_ID
    assert grant.operation_id == OPERATION
    assert grant.side_effect_scope_id != "evil"
    assert grant.side_effect_scope_digest != "evil_digest"
    assert grant.run_id != "evil_run"
    assert grant.operation_id != "evil"


def test_quote_acceptance_require_human_grant_preserves_canonical_digest() -> None:
    continuation = GovernedContinuationRequest(
        reason=ContinuationReason.QUOTE,
        task_id=TASK_ID,
        run_id=RUN_ID,
        source_agent_id="external_contractor_adapter",
        prompt="quote acceptance requires governed continuation",
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
        side_effect_scope_digest=QUOTE_SCOPE_DIGEST_1,
        operation_id="ACCEPT_QUOTE",
        policy_rule_id=POLICY_RULE,
        resource_scope=RESOURCE,
        policy_action=PolicyAction.REQUIRE_HUMAN,
        correlation={
            "quote_id": "EVIL",
            "quote_version": 999,
            "scope_digest": "evil_digest",
        },
        context={
            "quote_id": "EVIL",
            "quote_version": 999,
            "scope_digest": "evil_digest",
        },
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    _approve_resolution(
        task,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )
    grant = GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert grant is not None
    assert grant.side_effect_scope_digest == QUOTE_SCOPE_DIGEST_1
    assert grant.side_effect_scope_digest != SCOPE_DIGEST_1
    assert grant.side_effect_scope_digest != "evil_digest"


def test_new_pause_clears_stale_grant() -> None:
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    _approve_resolution(
        task,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )
    GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    assert task.runtime.governance.governed_continuation_grant is not None

    continuation_2 = _continuation_request(
        continuation_request_id=CONTINUATION_2,
        side_effect_scope_id=SCOPE_2,
    )
    _apply_governed_pause(task, continuation_2)
    assert task.runtime.governance.governed_continuation_grant is None


@pytest.mark.asyncio
async def test_intake_runner_approve_creates_grant_before_pause_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hitl_runtime_events(monkeypatch)
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    _set_human_response(
        task,
        response_text="approve",
        verdict=HumanResponseVerdict.APPROVE,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )

    runner, _published = _build_intake_runner_with_hitl()
    lifecycle = TaskLifecycle()
    trace_emitter = TaskTraceEmitter(run_id=RUN_ID, attempt_id=ATTEMPT_ID)
    with bound_hitl_test_execution_identity(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=EXECUTION_ID,
    ):
        await runner.run(task, lifecycle=lifecycle, trace_emitter=trace_emitter)

    grant = task.runtime.governance.governed_continuation_grant
    assert grant is not None
    assert grant.side_effect_scope_id == SCOPE_1
    assert task.runtime.governance.hitl_resolution is not None
    assert task.runtime.governance.hitl_resolution.verdict is HumanResponseVerdict.APPROVE


@pytest.mark.asyncio
async def test_intake_runner_reject_clears_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hitl_runtime_events(monkeypatch)
    continuation = _continuation_request(
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
    )
    task = _task()
    pause = _apply_governed_pause(task, continuation)
    task.runtime.governance.governed_continuation_grant = GovernedContinuationApprovalGrant(
        grant_id="gcg_stale",
        continuation_request_id=CONTINUATION_1,
        side_effect_scope_id=SCOPE_1,
        side_effect_scope_digest=None,
        task_id=TASK_ID,
        run_id=RUN_ID,
        operation_id=OPERATION,
        resource_scope=RESOURCE,
        policy_rule_id=POLICY_RULE,
        policy_bundle_id=BUNDLE_ID,
        policy_bundle_version=BUNDLE_VERSION,
        policy_bundle_digest=BUNDLE_DIGEST,
        pause_id=PAUSE_A,
        human_request_id=HR_A,
        approved_at="2026-08-18T00:00:00+00:00",
    )
    _set_human_response(
        task,
        response_text="reject",
        verdict=HumanResponseVerdict.REJECT,
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
    )

    runner, _published = _build_intake_runner_with_hitl()
    lifecycle = TaskLifecycle()
    with bound_hitl_test_execution_identity(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=EXECUTION_ID,
    ):
        outcome = await runner.run(task, lifecycle=lifecycle, trace_emitter=AsyncMock())
    assert outcome.early_result is not None
    assert task.runtime.governance.governed_continuation_grant is None
