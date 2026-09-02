# © Artur Czarnecki. All rights reserved.

"""Provider-neutral scenario execution through public authorization boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_lifecycle import TaskLifecycle

from tests.e2e.collaborative_work.harness.composition import MultiplayerE2EContext
from tests.e2e.collaborative_work.harness.constants import (
    OPERATION_MUTATE,
    PRINCIPAL_ALICE,
    RESOURCE_A,
    SIDE_EFFECT_DIGEST_1,
    SIDE_EFFECT_SCOPE_1,
    TENANT_A,
    WORKSPACE_A,
)
from tests.e2e.collaborative_work.harness.side_effect_probe import SideEffectProbe

T = TypeVar("T")


def allow_runtime_decision() -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="runtime allow",
        policy_rule_id="runtime.allow",
    )


def require_human_runtime_decision(
    *,
    policy_rule_id: str = "runtime.hitl",
    policy_bundle_id: str = "bundle-collab-e2e",
    policy_bundle_version: str = "1.0.0",
    policy_bundle_digest: str = "sha256:" + ("11" * 32),
) -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.REQUIRE_HUMAN,
        reason="runtime hitl",
        policy_rule_id=policy_rule_id,
        policy_bundle_id=policy_bundle_id,
        policy_bundle_version=policy_bundle_version,
        policy_bundle_digest=policy_bundle_digest,
    )


def build_enforcement_request(
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WORKSPACE_A,
    operation_id: str = OPERATION_MUTATE,
    acting_principal_id: str = PRINCIPAL_ALICE,
    delegator_principal_id: str | None = None,
    resource_scope: str = RESOURCE_A,
    task_id: str | None = None,
    run_id: str = "run-e2e-1",
    side_effect_scope_id: str = SIDE_EFFECT_SCOPE_1,
    side_effect_scope_digest: str | None = SIDE_EFFECT_DIGEST_1,
    side_effect_resource: str | None = None,
) -> CollaborativeWorkEnforcementRequest:
    resolved_task_id = task_id or mint_task_id()
    return CollaborativeWorkEnforcementRequest(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        operation_id=operation_id,
        acting_principal_id=acting_principal_id,
        delegator_principal_id=delegator_principal_id,
        resource_scope=resource_scope,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=operation_id,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id=side_effect_scope_id,
            side_effect_scope_digest=side_effect_scope_digest,
            task_id=resolved_task_id,
            run_id=run_id,
            principal_id=acting_principal_id,
            tenant_id=tenant_id,
            resource=side_effect_resource if side_effect_resource is not None else resource_scope,
        ),
    )


def run_multiplayer_e2e_scenario(
    context: MultiplayerE2EContext,
    request: CollaborativeWorkEnforcementRequest,
    probe: SideEffectProbe,
    *,
    task: Task | None = None,
    lifecycle: TaskLifecycle | None = None,
    on_authorization: Callable[[MeaningfulSideEffectAuthorizationResult], None] | None = None,
) -> object:
    side_effect = request.meaningful_side_effect_request
    assert side_effect is not None

    def execute() -> str:
        return probe.execute(side_effect)

    return context.boundary.authorize_and_execute(
        request,
        execute,
        task=task,
        lifecycle=lifecycle,
        on_authorization=on_authorization,
    )


def assert_authorization_denied(result: object, probe: SideEffectProbe) -> None:
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert probe.count == 0


def assert_authorization_allowed(result: object, probe: SideEffectProbe) -> None:
    assert result == "executed"
    assert probe.count == 1


def assert_require_human_paused(
    result: object,
    probe: SideEffectProbe,
    *,
    task: Task,
) -> None:
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.REQUIRE_HUMAN
    assert result.requires_governed_continuation is True
    assert probe.count == 0
    assert task.runtime.governance.paused is True
