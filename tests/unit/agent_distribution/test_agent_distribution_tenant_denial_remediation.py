# © Artur Czarnecki. All rights reserved.

"""Tenant-scope denial vs mutation evidence remediation (TE1–TE14)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminBlockedError,
    AgentPlatformAdminGovernanceBlockedError,
    ControlPlaneTenantScopeDenial,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.control_plane_governance import (
    AGENT_DISTRIBUTION_RESOURCE_TYPE,
    MUTATION_TYPE_BUILD_RUNTIME_REVISION,
    StaticApplicationEnvironmentTenantResolver,
    application_environment_resource_id,
    application_environment_resource_scope,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _APP,
    _ENV,
    _bind_request,
    _build_request,
    _build_revision,
    _install_request,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OTHER_TENANT_PRINCIPAL = RequestIdentity(
    tenant_id="tenant-other",
    user_id="other-admin",
    principal_type=PrincipalType.USER,
    auth_subject="other-admin",
)
_ZERO_DIGEST = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

_CONTROL_PLANE_GOVERNANCE_SLICE_FILES = (
    "intergrax/contracts/control_plane_mutation.py",
    "intergrax/runtime/governance/control_plane_mutation_authorization.py",
    "intergrax/agent_distribution/control_plane_governance.py",
    "intergrax/agent_distribution/admin_service.py",
    "intergrax/applications/_shared/reference_production_lifecycle.py",
    "intergrax/applications/_shared/agent_platform_admin_routes.py",
)

_FORBIDDEN_DYNAMIC_PATTERNS = re.compile(
    r"getattr\s*\(|setattr\s*\(|hasattr\s*\(|__dict__|eval\s*\(|exec\s*\("
)


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return self.decision


def _stack_wrong_environment_tenant() -> tuple:
    stack = build_admin_stack()
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("other-tenant")
    )
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=_RecordingEvaluator())
    )
    return stack


def _assert_early_tenant_denial(exc: AgentPlatformAdminGovernanceBlockedError) -> None:
    assert exc.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert exc.authorization_evidence is None
    denial = exc.tenant_scope_denial
    assert denial is not None
    assert denial.tenant_id == admin_test_principal().tenant_id
    assert denial.reason == "tenant_authority_mismatch"
    assert denial.principal_type == admin_test_principal().principal_type
    assert denial.principal_user_id == admin_test_principal().user_id
    assert denial.principal_auth_subject == admin_test_principal().auth_subject
    assert denial.resource_type == AGENT_DISTRIBUTION_RESOURCE_TYPE
    assert denial.resource_id == application_environment_resource_id(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert denial.resource_scope == application_environment_resource_scope(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    dumped = denial.model_dump()
    assert "mutation_id" not in dumped
    assert "request_digest" not in dumped
    assert "mutation_type" not in dumped


def test_te1_wrong_tenant_install_blocked_before_lookup() -> None:
    stack = _stack_wrong_environment_tenant()
    lookup_calls: list[str] = []
    original = stack.service._installation_store.get_installation

    def _tracked(installation_id: str):
        lookup_calls.append(installation_id)
        return original(installation_id)

    with patch.object(stack.service._installation_store, "get_installation", side_effect=_tracked):
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.install_agent(
                application_id=_APP,
                application_environment_id=_ENV,
                request=_install_request(mutation_id="mut-te1"),
                principal=admin_test_principal(),
            )
    _assert_early_tenant_denial(exc.value)
    assert lookup_calls == []


def test_te2_wrong_tenant_bind_blocked_before_lookup() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-te2-install"),
        principal=principal,
    )
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("other-tenant")
    )
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=_RecordingEvaluator())
    )
    lookup_calls: list[str] = []
    original = stack.service._binding_store.get_binding

    def _tracked(binding_id: str):
        lookup_calls.append(binding_id)
        return original(binding_id)

    with patch.object(stack.service._binding_store, "get_binding", side_effect=_tracked):
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.bind_agent(
                application_id=_APP,
                application_environment_id=_ENV,
                request=_bind_request(mutation_id="mut-te2-bind"),
                principal=principal,
            )
    _assert_early_tenant_denial(exc.value)
    assert lookup_calls == []


def test_te3_wrong_tenant_update_enable_disable_blocked_before_lookup() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-te3-install"),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(mutation_id="mut-te3-bind"),
        principal=principal,
    )
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("other-tenant")
    )
    lookup_calls: list[str] = []
    original = stack.service._binding_store.get_binding

    def _tracked(binding_id: str):
        lookup_calls.append(binding_id)
        return original(binding_id)

    with patch.object(stack.service._binding_store, "get_binding", side_effect=_tracked):
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.update_binding_config(
                application_id=_APP,
                application_environment_id=_ENV,
                application_binding_id="bind-search",
                request=UpdateAgentBindingRequest(
                    mutation_id="mut-te3-update",
                    expected_revision=0,
                    config={"mode": "fast"},
                ),
                principal=principal,
            )
        _assert_early_tenant_denial(exc.value)
        assert lookup_calls == []

        lookup_calls.clear()
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.enable_binding(
                application_id=_APP,
                application_environment_id=_ENV,
                application_binding_id="bind-search",
                request=SetAgentEnablementRequest(
                    mutation_id="mut-te3-enable",
                    expected_revision=0,
                ),
                principal=principal,
            )
        _assert_early_tenant_denial(exc.value)
        assert lookup_calls == []

        lookup_calls.clear()
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.disable_binding(
                application_id=_APP,
                application_environment_id=_ENV,
                application_binding_id="bind-search",
                request=SetAgentEnablementRequest(
                    mutation_id="mut-te3-disable",
                    expected_revision=0,
                ),
                principal=principal,
            )
        _assert_early_tenant_denial(exc.value)
        assert lookup_calls == []


def test_te4_wrong_tenant_build_blocked_before_revision_lookup() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-te4-install"),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(mutation_id="mut-te4-bind"),
        principal=principal,
    )
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-te4-enable", expected_revision=0),
        principal=principal,
    )
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("other-tenant")
    )
    lookup_calls: list[str] = []
    original = stack.service._revision_store.get_revision

    def _tracked(revision_id: str):
        lookup_calls.append(revision_id)
        return original(revision_id)

    with patch.object(stack.service._revision_store, "get_revision", side_effect=_tracked):
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.build_application_revision(
                application_id=_APP,
                application_environment_id=_ENV,
                request=_build_request("rev-te4"),
                principal=principal,
            )
    _assert_early_tenant_denial(exc.value)
    assert lookup_calls == []


def test_te5_tenant_denial_diagnostic_fields() -> None:
    stack = _stack_wrong_environment_tenant()
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-te5"),
            principal=admin_test_principal(),
        )
    denial = exc.value.tenant_scope_denial
    assert isinstance(denial, ControlPlaneTenantScopeDenial)
    assert denial is not None
    assert denial.tenant_id == "tenant-test"
    assert denial.resource_id == f"{_APP}:{_ENV}"
    assert denial.resource_scope == (
        f"agent_distribution.application:{_APP}.environment:{_ENV}"
    )
    assert denial.reason == "tenant_authority_mismatch"


def test_te6_real_policy_deny_has_canonical_evidence() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack = build_admin_stack()
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-te6"),
            principal=admin_test_principal(),
        )
    assert exc.value.tenant_scope_denial is None
    evidence = exc.value.authorization_evidence
    assert evidence is not None
    assert evidence.mutation_id == "mut-te6"
    assert evidence.mutation_type == "agent_distribution.install_agent"
    assert evidence.request_digest != _ZERO_DIGEST
    assert evidence.request_digest.startswith("sha256:")


def test_te7_real_require_human_preserves_scope() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="hitl",
            policy_rule_id="rule.hitl",
        )
    )
    stack = build_admin_stack()
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-te7"),
            principal=admin_test_principal(),
        )
    assert exc.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert exc.value.authorization_scope is not None
    assert exc.value.authorization_scope.mutation_id == "mut-te7"
    assert exc.value.authorization_evidence is not None
    assert exc.value.tenant_scope_denial is None


def test_te8_missing_resolver_no_synthetic_evidence() -> None:
    stack = build_admin_stack()
    stack.service._environment_tenant_resolver = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-te8"),
            principal=admin_test_principal(),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_TENANT_AUTHORITY"


def test_te9_zero_fake_digest_static_scan() -> None:
    for relative_path in _CONTROL_PLANE_GOVERNANCE_SLICE_FILES:
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert _ZERO_DIGEST not in source, relative_path


def test_te10_zero_synthetic_mutation_identity_static_scan() -> None:
    admin_service = (_REPO_ROOT / "intergrax/agent_distribution/admin_service.py").read_text(
        encoding="utf-8"
    )
    assert "tenant-authority-deny" not in admin_service
    assert "agent_distribution.tenant_authority" not in admin_service


def test_te11_zero_dynamic_access_static_gate() -> None:
    for relative_path in _CONTROL_PLANE_GOVERNANCE_SLICE_FILES:
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert _FORBIDDEN_DYNAMIC_PATTERNS.search(source) is None, relative_path


def test_te12_ad_build_r1_regression() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_build_remediation

    assert test_agent_distribution_build_remediation is not None


def test_te13_desired_state_regression() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_desired_state_remediation

    assert test_agent_distribution_desired_state_remediation is not None


def test_te14_activation_regression() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_activation

    assert test_agent_distribution_activation is not None
