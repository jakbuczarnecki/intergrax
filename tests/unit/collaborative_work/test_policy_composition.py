# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1E — fail-closed policy composition tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.policy_composition import compose_policy_decisions
from intergrax.collaborative_work.repository import (
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    EffectiveAuthorityRequest,
    MembershipStatus,
    PolicyCompositionApplicability,
    PolicyCompositionInput,
    PolicyCompositionLayer,
    PolicyCompositionResult,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_SCOPE = "collab:mutate"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _allow(rule_id: str = "test.allow") -> PolicyDecision:
    return PolicyDecision(action=PolicyAction.ALLOW, reason="allow", policy_rule_id=rule_id)


def _deny(rule_id: str = "test.deny", reason: str = "deny") -> PolicyDecision:
    return PolicyDecision(action=PolicyAction.DENY, reason=reason, policy_rule_id=rule_id)


def _require_human(rule_id: str = "test.require_human") -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.REQUIRE_HUMAN,
        reason="require_human",
        policy_rule_id=rule_id,
    )


def _escalate(rule_id: str = "test.escalate") -> PolicyDecision:
    return PolicyDecision(action=PolicyAction.ESCALATE, reason="escalate", policy_rule_id=rule_id)


def _modify(rule_id: str = "test.modify") -> PolicyDecision:
    return PolicyDecision(action=PolicyAction.MODIFY, reason="modify", policy_rule_id=rule_id)


def _all_mandatory_applicability() -> PolicyCompositionApplicability:
    return PolicyCompositionApplicability(
        workspace_policy=PolicyLayerApplicability.REQUIRED,
        resource_policy=PolicyLayerApplicability.REQUIRED,
        runtime_policy=PolicyLayerApplicability.REQUIRED,
    )


def _not_applicable_applicability() -> PolicyCompositionApplicability:
    return PolicyCompositionApplicability(
        workspace_policy=PolicyLayerApplicability.NOT_APPLICABLE,
        resource_policy=PolicyLayerApplicability.NOT_APPLICABLE,
        runtime_policy=PolicyLayerApplicability.NOT_APPLICABLE,
    )


def _compose(
    *,
    collaborative: PolicyDecision,
    workspace: PolicyDecision | None = None,
    resource: PolicyDecision | None = None,
    runtime: PolicyDecision | None = None,
    applicability: PolicyCompositionApplicability | None = None,
) -> PolicyCompositionResult:
    return compose_policy_decisions(
        PolicyCompositionInput(
            collaborative_authority=collaborative,
            workspace_policy=workspace,
            resource_policy=resource,
            runtime_policy=runtime,
            applicability=applicability or _all_mandatory_applicability(),
        )
    )


def test_all_mandatory_layers_allow() -> None:
    result = _compose(
        collaborative=_allow("collab.allow"),
        workspace=_allow("workspace.allow"),
        resource=_allow("resource.allow"),
        runtime=_allow("runtime.allow"),
    )
    assert result.decision.action is PolicyAction.ALLOW
    assert result.determining_layer is None
    assert result.decision.audit_payload["non_allow_layers"] == []


def test_collaborative_deny_with_all_others_allow() -> None:
    result = _compose(
        collaborative=_deny("collab.deny"),
        workspace=_allow(),
        resource=_allow(),
        runtime=_allow(),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.COLLABORATIVE_AUTHORITY


def test_workspace_deny() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_deny("workspace.deny"),
        resource=_allow(),
        runtime=_allow(),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.WORKSPACE_POLICY


def test_resource_deny() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=_deny("resource.deny"),
        runtime=_allow(),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.RESOURCE_POLICY


def test_runtime_deny() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=_allow(),
        runtime=_deny("runtime.deny"),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.RUNTIME_POLICY


def test_runtime_require_human_survives_composition() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=_allow(),
        runtime=_require_human("runtime.hitl"),
    )
    assert result.decision.action is PolicyAction.REQUIRE_HUMAN
    assert result.determining_layer is PolicyCompositionLayer.RUNTIME_POLICY


def test_missing_mandatory_workspace_decision() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=None,
        resource=_allow(),
        runtime=_allow(),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.WORKSPACE_POLICY
    assert "missing" in result.decision.reason


def test_missing_mandatory_resource_decision_when_required() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=None,
        runtime=_allow(),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.RESOURCE_POLICY


def test_missing_runtime_decision_for_meaningful_side_effect() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=_allow(),
        runtime=None,
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.RUNTIME_POLICY


def test_no_layer_overrides_prior_deny() -> None:
    result = _compose(
        collaborative=_deny("collab.deny"),
        workspace=_allow(),
        resource=_allow(),
        runtime=_deny("runtime.deny"),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.COLLABORATIVE_AUTHORITY


def test_modify_conservative_fail_closed() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_modify(),
        resource=_allow(),
        runtime=_allow(),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.WORKSPACE_POLICY


def test_escalate_preserved() -> None:
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=_escalate("resource.escalate"),
        runtime=_allow(),
    )
    assert result.decision.action is PolicyAction.ESCALATE
    assert result.determining_layer is PolicyCompositionLayer.RESOURCE_POLICY


def test_not_applicable_layers_skipped() -> None:
    result = compose_policy_decisions(
        PolicyCompositionInput(
            collaborative_authority=_allow(),
            applicability=_not_applicable_applicability(),
        )
    )
    assert result.decision.action is PolicyAction.ALLOW


def test_default_applicability_fails_closed() -> None:
    result = compose_policy_decisions(
        PolicyCompositionInput(collaborative_authority=_allow()),
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.WORKSPACE_POLICY
    assert (
        result.decision.policy_rule_id
        == "collaborative_work.policy_composition.applicability_unresolved.workspace_policy"
    )


@pytest.mark.parametrize(
    ("layer_field", "determining_layer"),
    [
        ("workspace_policy", PolicyCompositionLayer.WORKSPACE_POLICY),
        ("resource_policy", PolicyCompositionLayer.RESOURCE_POLICY),
        ("runtime_policy", PolicyCompositionLayer.RUNTIME_POLICY),
    ],
)
def test_unknown_applicability_fails_closed(
    layer_field: str,
    determining_layer: PolicyCompositionLayer,
) -> None:
    applicability = _all_mandatory_applicability().model_copy(
        update={layer_field: PolicyLayerApplicability.UNKNOWN},
    )
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=_allow(),
        runtime=_allow(),
        applicability=applicability,
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is determining_layer
    assert (
        result.decision.policy_rule_id
        == f"collaborative_work.policy_composition.applicability_unresolved.{determining_layer.value}"
    )


def test_unresolved_applicability_audit_provenance() -> None:
    result = compose_policy_decisions(
        PolicyCompositionInput(collaborative_authority=_allow()),
    )
    assert result.decision.audit_payload["determining_layer"] == "workspace_policy"
    assert (
        result.decision.audit_payload["determining_layer_audit"]["cause"]
        == "applicability_unresolved"
    )
    assert "workspace_policy" in result.decision.audit_payload["non_allow_layers"]


def test_audit_provenance_retained() -> None:
    workspace = _deny("workspace.deny", reason="workspace blocked")
    result = _compose(
        collaborative=_allow("collab.allow"),
        workspace=workspace,
        resource=_allow("resource.allow"),
        runtime=_allow("runtime.allow"),
    )
    layers = result.decision.audit_payload["contributing_layers"]
    assert layers["collaborative_authority"]["action"] == PolicyAction.ALLOW.value
    assert layers["workspace_policy"]["reason"] == "workspace blocked"
    assert result.decision.audit_payload["determining_layer"] == "workspace_policy"
    assert "workspace_policy" in result.decision.audit_payload["non_allow_layers"]


def test_policy_bundle_not_fabricated() -> None:
    bundled = PolicyDecision(
        action=PolicyAction.DENY,
        reason="bundled deny",
        policy_rule_id="runtime.bundled",
        policy_bundle_id="bundle-1",
        policy_bundle_version="1",
        policy_bundle_digest="sha256:abc",
    )
    result = _compose(
        collaborative=_allow(),
        workspace=_allow(),
        resource=_allow(),
        runtime=bundled,
    )
    assert result.decision.policy_bundle_id == "bundle-1"
    assert result.decision.policy_bundle_version == "1"
    assert result.decision.policy_bundle_digest == "sha256:abc"


def test_runtime_meaningful_side_effect_deny_survives_composition() -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="authority-grant-1",
            principal_id=_ACTING,
            authority_scopes=(_SCOPE,),
        )
    )
    membership_locator = WorkspaceMembership.model_validate(
        {
            "membership_id": "membership-1",
            "tenant_id": _TENANT,
            "workspace_id": _WORKSPACE,
            "principal_id": _ACTING,
            "role": WorkspaceMembershipRole.MEMBER,
            "status": MembershipStatus.ACTIVE,
            "revision": 0,
        }
    )
    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=authority_repo,
        clock=lambda: _NOW,
    )
    authority = resolver.resolve(
        EffectiveAuthorityRequest(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            acting_principal_id=_ACTING,
            requested_authority_scopes=(_SCOPE,),
            membership=membership_locator,
        )
    )
    assert authority.decision.action is PolicyAction.ALLOW

    runtime_engine = RuntimePolicyEngine(rules=[])
    side_effect_request = MeaningfulSideEffectRequest(
        action="ACCEPT_QUOTE",
        kinds=(MeaningfulSideEffectKind.COMMITMENT,),
        task_id="task-1",
        run_id="run-1",
        principal_id=_ACTING,
        tenant_id=_TENANT,
    )
    runtime_decision = runtime_engine.evaluate_meaningful_side_effect(side_effect_request)
    assert runtime_decision.action is PolicyAction.DENY

    result = compose_policy_decisions(
        PolicyCompositionInput(
            collaborative_authority=authority.decision,
            workspace_policy=_allow("workspace.allow"),
            resource_policy=_allow("resource.allow"),
            runtime_policy=runtime_decision,
            applicability=_all_mandatory_applicability(),
        )
    )
    assert result.decision.action is PolicyAction.DENY
    assert result.determining_layer is PolicyCompositionLayer.RUNTIME_POLICY
