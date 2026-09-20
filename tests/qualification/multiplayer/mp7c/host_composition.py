# © Artur Czarnecki. All rights reserved.

"""Composition-only fixture: exercise canonical harness-host MSE wiring (MP-7C).

Private Collaborative Work and host-wiring imports stay in this module.
The Tier-3 consumer never imports this graph — it receives only the public Protocol.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from unittest.mock import patch

from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    HarnessMeaningfulSideEffectAuthorizationWiring,
    resolve_harness_host_meaningful_side_effect_authorization_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.collaborative_work.enforcement_gate import (
    MeaningfulSideEffectPolicyEvaluator,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    CollaborativeWorkEnforcementRequest,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.meaningful_side_effect_policy import (
    MeaningfulSideEffectPolicyRule,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

_TENANT = "mp7c-tenant"
_WORKSPACE = "mp7c-workspace"
_ACTING = "mp7c-principal"
_OPERATION = "mp7c.proof.mutation"
_SCOPE = "mp7c.proof.scope"
_RESOURCE = "mp7c-resource-1"


class _ObservableCollaborativeWorkStore:
    """Test store owner used only to observe lifecycle/close semantics."""

    def __init__(self) -> None:
        self.closed = False
        self.materialization_token = "mp7c-in-memory-store"

    def close(self) -> None:
        self.closed = True


def strict_host_environment() -> ApplicationEnvironmentProfile:
    """Minimal strict Tier-3 host environment for composition proofs."""
    base = ApplicationEnvironmentProfile()
    return base.model_copy(
        update={
            "meta": base.meta.model_copy(
                update={"execution_mode": ExecutionMode.STRICT}
            ),
        },
    )


def non_strict_host_environment() -> ApplicationEnvironmentProfile:
    """Non-strict host — canonical wiring returns authorization_port=None."""
    base = ApplicationEnvironmentProfile()
    return base.model_copy(
        update={
            "meta": base.meta.model_copy(
                update={"execution_mode": ExecutionMode.BALANCED}
            ),
        },
    )


def empty_in_memory_repositories(
    store: _ObservableCollaborativeWorkStore | None = None,
) -> CollaborativeWorkRepositories:
    """Caller-owned in-memory CW bundle (no provider resolution)."""
    return CollaborativeWorkRepositories(
        membership=InMemoryWorkspaceMembershipRepository(),
        delegation=InMemoryAuthorityDelegationRepository(),
        principal_authority=InMemoryPrincipalAuthorityRepository(),
        policy=InMemoryCollaborativePolicyRepository(),
        operation_profile=InMemoryCollaborativeOperationPolicyProfileRepository(),
        store=store if store is not None else _ObservableCollaborativeWorkStore(),
    )


def resolve_host_wiring(
    environment: ApplicationEnvironmentProfile,
    *,
    explicit: MeaningfulSideEffectAuthorizationPort | None = None,
    collaborative_work_repositories: CollaborativeWorkRepositories | None = None,
    collaborative_work_integration_profile: IntegrationProfile | None = None,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
) -> HarnessMeaningfulSideEffectAuthorizationWiring:
    """Call the canonical host resolver (optionally with injected runtime policy evaluator).

    Empty ``RuntimePolicyEngine()`` remains the production default when no evaluator
    is injected (fail-closed on indeterminate MSE runtime policy).
    """
    return resolve_harness_host_meaningful_side_effect_authorization_wiring(
        environment,
        explicit=explicit,
        collaborative_work_repositories=collaborative_work_repositories,
        collaborative_work_integration_profile=collaborative_work_integration_profile,
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        runtime_policy_evaluator=runtime_policy_evaluator,
    )


@dataclass(frozen=True, slots=True)
class HostAuthorizationScenario:
    """Resolved host port + public enforcement request + active-execution bind helpers."""

    wiring: HarnessMeaningfulSideEffectAuthorizationWiring
    request: CollaborativeWorkEnforcementRequest
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    repositories: CollaborativeWorkRepositories | None = None

    @property
    def authorization_port(self) -> MeaningfulSideEffectAuthorizationPort:
        port = self.wiring.authorization_port
        if port is None:
            raise AssertionError(
                "expected host-resolved MeaningfulSideEffectAuthorizationPort"
            )
        return port


def _allow_runtime_rules() -> tuple[MeaningfulSideEffectPolicyRule, ...]:
    return (
        MeaningfulSideEffectPolicyRule(
            rule_id="mp7c.runtime.allow",
            action=_OPERATION,
            decision=PolicyAction.ALLOW,
        ),
    )


def _seed_allow_state(
    bundle: CollaborativeWorkRepositories,
) -> WorkspaceMembership:
    membership = bundle.membership.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="mp7c-membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    bundle.principal_authority.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="mp7c-grant-1",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    bundle.policy.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="mp7c-ws-allow",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    bundle.policy.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="mp7c-res-allow",
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope=_SCOPE,
            resource_scope=_RESOURCE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    bundle.operation_profile.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=_OPERATION,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_policy_applicability=PolicyLayerApplicability.REQUIRED,
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=OperationPolicyRequirement.REQUIRED,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )
    return WorkspaceMembership.model_validate(membership.model_dump())


def _build_request(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    membership: WorkspaceMembership | None,
) -> CollaborativeWorkEnforcementRequest:
    return CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_ACTING,
        resource_scope=_RESOURCE,
        membership=membership,
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=_OPERATION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="mp7c-side-effect-scope",
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            principal_id=_ACTING,
            tenant_id=_TENANT,
            resource=_RESOURCE,
        ),
    )


@contextmanager
def bound_host_active_execution(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> Iterator[None]:
    """Register process-local active task scope + bind execution identity (host default guard)."""
    ActiveTaskRegistry.clear_for_tests()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_ACTING,
        message="mp7c-host-composition",
        context=TaskContext(),
    )
    asyncio.run(ActiveTaskRegistry.register(task, run_id))
    try:
        with bound_gr3_active_execution(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ):
            yield
    finally:
        ActiveTaskRegistry.clear_for_tests()


def compose_host_default_deny_scenario() -> HostAuthorizationScenario:
    """Strict host + injected empty CW repos → real platform DENY via host resolver."""
    bundle = empty_in_memory_repositories()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    wiring = resolve_host_wiring(
        strict_host_environment(),
        collaborative_work_repositories=bundle,
    )
    fake_membership = WorkspaceMembership(
        membership_id="mp7c-fake-membership",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_ACTING,
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
        revision=0,
    )
    request = _build_request(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        membership=fake_membership,
    )
    return HostAuthorizationScenario(
        wiring=wiring,
        request=request,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        repositories=bundle,
    )


def compose_host_default_allow_scenario() -> HostAuthorizationScenario:
    """Strict host + seeded CW repos + injected runtime evaluator → real platform ALLOW."""
    bundle = empty_in_memory_repositories()
    membership = _seed_allow_state(bundle)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    wiring = resolve_host_wiring(
        strict_host_environment(),
        collaborative_work_repositories=bundle,
        runtime_policy_evaluator=RuntimePolicyEngine(
            meaningful_side_effect_rules=_allow_runtime_rules(),
        ),
    )
    request = _build_request(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        membership=membership,
    )
    return HostAuthorizationScenario(
        wiring=wiring,
        request=request,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        repositories=bundle,
    )


def compose_host_materialized_default_wiring(
    *,
    resolve_repositories: Callable[[IntegrationProfile], CollaborativeWorkRepositories],
    collaborative_work_integration_profile: IntegrationProfile | None = None,
    runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None = None,
) -> HarnessMeaningfulSideEffectAuthorizationWiring:
    """Strict host without injected repos — resolver materializes via provider selection."""
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
        side_effect=resolve_repositories,
    ):
        return resolve_host_wiring(
            strict_host_environment(),
            collaborative_work_integration_profile=collaborative_work_integration_profile,
            runtime_policy_evaluator=runtime_policy_evaluator,
        )


def compose_host_default_fail_closed_no_runtime_rule_scenario() -> (
    HostAuthorizationScenario
):
    """Strict host + allow-ish CW state + default empty evaluator → not ALLOW (fail closed)."""
    bundle = empty_in_memory_repositories()
    membership = _seed_allow_state(bundle)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    wiring = resolve_host_wiring(
        strict_host_environment(),
        collaborative_work_repositories=bundle,
    )
    request = _build_request(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        membership=membership,
    )
    return HostAuthorizationScenario(
        wiring=wiring,
        request=request,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        repositories=bundle,
    )


__all__ = [
    "HostAuthorizationScenario",
    "_ObservableCollaborativeWorkStore",
    "bound_host_active_execution",
    "compose_host_default_allow_scenario",
    "compose_host_default_deny_scenario",
    "compose_host_default_fail_closed_no_runtime_rule_scenario",
    "compose_host_materialized_default_wiring",
    "empty_in_memory_repositories",
    "non_strict_host_environment",
    "resolve_host_wiring",
    "strict_host_environment",
]
