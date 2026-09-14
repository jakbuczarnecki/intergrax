# © Artur Czarnecki. All rights reserved.

"""GR-2 enterprise proofs — contract-first root execution admission governance."""

from __future__ import annotations

import pytest

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionRequest,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RootExecutionAdmissionPolicyRule,
    RuntimeExecutionPolicyAdmissionPort,
    RuntimeExecutionPolicyAdmissionRequest,
    RuntimeExecutionPolicyAdmissionResult,
    WORKER_ROOT_EXECUTION_OPERATION,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.execution_admission_composition import (
    build_fail_closed_runtime_execution_policy_admission,
    build_root_execution_authority_admission,
    build_root_execution_authority_admission_from_rules,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
    RuntimeExecutionPolicyAdmissionEvaluator,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit


class _StaticRuntimePolicyAdmission(RuntimeExecutionPolicyAdmissionPort):
    def __init__(self, *, decision: PolicyAction) -> None:
        self._decision = decision

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=self._decision,
                reason=f"static.{self._decision.value}",
                policy_rule_id=f"test.static.{self._decision.value}",
            ),
        )


def _authorize_with_port(
    port: RuntimeExecutionPolicyAdmissionPort,
    *,
    scopes: tuple[str, ...] = ("workspace.read", "workspace.write"),
) -> RootExecutionAuthorityAdmissionDisposition:
    service = build_root_execution_authority_admission(runtime_policy_admission=port)
    result = service.authorize(
        RootExecutionAuthorityAdmissionRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-x",
            principal_id="principal-1",
            collaborative_authority_scopes=scopes,
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="collaborative"),
            ),
        )
    )
    return result.disposition


def test_gr2_pluggable_policy_port_allow_vs_deny_without_consumer_change() -> None:
    assert (
        _authorize_with_port(_StaticRuntimePolicyAdmission(decision=PolicyAction.ALLOW))
        is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    )
    assert (
        _authorize_with_port(_StaticRuntimePolicyAdmission(decision=PolicyAction.DENY))
        is RootExecutionAuthorityAdmissionDisposition.DENIED
    )


def test_gr2_fail_closed_when_composition_uses_unconfigured_policy_engine() -> None:
    service = build_root_execution_authority_admission(
        runtime_policy_admission=build_fail_closed_runtime_execution_policy_admission(),
    )
    result = service.authorize(
        RootExecutionAuthorityAdmissionRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-x",
            principal_id="principal-1",
            collaborative_authority_scopes=("workspace.read",),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="collaborative"),
            ),
        )
    )
    assert result.disposition in {
        RootExecutionAuthorityAdmissionDisposition.DENIED,
        RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE,
    }
    assert result.trusted_parent_execution_authority is None


def test_gr2_scope_narrowing_read_write_to_read() -> None:
    service = build_root_execution_authority_admission_from_rules(
        root_execution_admission_rules=(
            RootExecutionAdmissionPolicyRule(
                rule_id="runtime.read_only",
                decision=PolicyAction.ALLOW,
                execution_operation=WORKER_ROOT_EXECUTION_OPERATION,
                approved_scopes=("workspace.read",),
            ),
        ),
    )
    result = service.authorize(
        RootExecutionAuthorityAdmissionRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-x",
            principal_id="principal-1",
            collaborative_authority_scopes=("workspace.read", "workspace.write"),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="collaborative"),
            ),
        )
    )
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert result.trusted_parent_execution_authority is not None
    assert result.trusted_parent_execution_authority.permission_scopes == ("workspace.read",)


def test_gr2_policy_cannot_widen_authority_with_admin_scope() -> None:
    widening_port = RuntimeExecutionPolicyAdmissionEvaluator(
        policy_engine=RuntimePolicyEngine(
            root_execution_admission_rules=(
                RootExecutionAdmissionPolicyRule(
                    rule_id="runtime.admin_escape",
                    decision=PolicyAction.ALLOW,
                    execution_operation=WORKER_ROOT_EXECUTION_OPERATION,
                    approved_scopes=("workspace.admin",),
                ),
            ),
        ),
    )
    service = build_root_execution_authority_admission(
        runtime_policy_admission=widening_port,
    )
    result = service.authorize(
        RootExecutionAuthorityAdmissionRequest(
            tenant_id="tenant-a",
            workspace_id="workspace-x",
            principal_id="principal-1",
            collaborative_authority_scopes=("workspace.read", "workspace.write"),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="collaborative"),
            ),
        )
    )
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED
    assert result.policy_decision is not None
    assert (
        result.policy_decision.reason
        == "runtime_approved_scopes_exceed_collaborative_authority"
    )
    assert result.trusted_parent_execution_authority is None


def test_gr2_composition_selects_allowing_vs_denying_adapters() -> None:
    allow_service = build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    deny_service = build_root_execution_authority_admission(
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
    )
    request = RootExecutionAuthorityAdmissionRequest(
        tenant_id="tenant-a",
        workspace_id="workspace-x",
        principal_id="principal-1",
        collaborative_authority_scopes=("workspace.read",),
        effective_authority_decision=EffectiveAuthorityDecision(
            decision=PolicyDecision(action=PolicyAction.ALLOW, reason="collaborative"),
        ),
    )
    assert (
        allow_service.authorize(request).disposition
        is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    )
    assert (
        deny_service.authorize(request).disposition
        is RootExecutionAuthorityAdmissionDisposition.DENIED
    )
