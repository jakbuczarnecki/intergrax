# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R2 — runtime physical delegation governance tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import GovernanceEvaluationPoint
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationCapabilityRequirement,
    PhysicalDelegationGovernancePolicyRule,
    PhysicalDelegationGovernanceRequest,
    PhysicalDelegationGovernedContinuation,
    PhysicalDelegationSelectedIdentity,
    build_physical_delegation_governed_continuation,
    physical_delegation_governance_request_digest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.physical_delegation_governance import (
    AllowingPhysicalDelegationGovernance,
    DenyingPhysicalDelegationGovernance,
    RequireHumanPhysicalDelegationGovernance,
    RuntimePhysicalDelegationGovernance,
    UnavailablePhysicalDelegationGovernance,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit


def _selected_identity(package_id: str = "ocr-agent") -> PhysicalDelegationSelectedIdentity:
    return PhysicalDelegationSelectedIdentity(
        catalog_source_id="builtin-1",
        provider_kind="builtin",
        distribution_package_id=package_id,
        package_version="1.0.0",
        package_digest="sha256:abc",
    )


def _request(
    *,
    package_id: str = "ocr-agent",
) -> PhysicalDelegationGovernanceRequest:
    return PhysicalDelegationGovernanceRequest(
        delegation_id="delegation-1",
        task_scope_id="task-scope-1",
        application_id="app-a",
        application_environment_id="env-a",
        principal=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            auth_subject="subject-1",
        ),
        capability_requirement=PhysicalDelegationCapabilityRequirement(
            required_capability_ids=("document.ocr",),
        ),
        selected_identity=_selected_identity(package_id),
    )


def test_governance_request_uses_multi_agent_delegation_evaluation_point() -> None:
    request = _request()
    assert (
        request.evaluation_point is GovernanceEvaluationPoint.MULTI_AGENT_DELEGATION
    )


def test_request_digest_is_sha256() -> None:
    digest = physical_delegation_governance_request_digest(_request())
    assert digest.startswith("sha256:")


def test_allowing_adapter_permits_without_widening_authority() -> None:
    request = _request()
    before = request.model_dump(mode="json")
    result = AllowingPhysicalDelegationGovernance().evaluate(request)
    after = request.model_dump(mode="json")
    assert before == after
    assert result.permitted is True
    assert result.decision.action is PolicyAction.ALLOW


def test_denying_adapter_denies() -> None:
    result = DenyingPhysicalDelegationGovernance().evaluate(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_require_human_adapter_requires_continuation() -> None:
    result = RequireHumanPhysicalDelegationGovernance().evaluate(_request())
    assert result.permitted is False
    assert result.requires_governed_continuation is True
    assert result.decision.action is PolicyAction.REQUIRE_HUMAN


def test_governed_continuation_binds_exact_selected_identity() -> None:
    request = _request(package_id="ocr-agent")
    result = RequireHumanPhysicalDelegationGovernance().evaluate(request)
    continuation = build_physical_delegation_governed_continuation(
        request=request,
        governance_result=result,
    )
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)
    assert continuation.delegation_id == "delegation-1"
    assert continuation.selected_identity.distribution_package_id == "ocr-agent"
    assert continuation.governance_result is result


def test_governed_continuation_rejects_inconsistent_contract() -> None:
    request = _request()
    allowed = AllowingPhysicalDelegationGovernance().evaluate(request)
    with pytest.raises(ValueError, match="must not be permitted"):
        build_physical_delegation_governed_continuation(
            request=request,
            governance_result=allowed,
        )


def test_unavailable_adapter_fail_closed() -> None:
    result = UnavailablePhysicalDelegationGovernance().evaluate(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_runtime_evaluator_allow_by_package() -> None:
    governance = RuntimePhysicalDelegationGovernance(
        policy_engine=RuntimePolicyEngine(
            physical_delegation_rules=(
                PhysicalDelegationGovernancePolicyRule(
                    rule_id="delegation.ocr.allow",
                    decision=PolicyAction.ALLOW,
                    package_id="ocr-agent",
                ),
                PhysicalDelegationGovernancePolicyRule(
                    rule_id="delegation.legal.deny",
                    decision=PolicyAction.DENY,
                    package_id="legal-agent",
                ),
            ),
        ),
    )
    allowed = governance.evaluate(_request(package_id="ocr-agent"))
    denied = governance.evaluate(_request(package_id="legal-agent"))
    assert allowed.permitted is True
    assert denied.permitted is False
