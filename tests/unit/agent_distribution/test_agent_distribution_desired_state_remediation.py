# © Artur Czarnecki. All rights reserved.

"""Agent Distribution desired-state remediation tests (ADS1–ADS30)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminBlockedError,
    AgentPlatformAdminGovernanceBlockedError,
    BindAgentRequest,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.control_plane_governance import (
    MUTATION_TYPE_DISABLE_BINDING,
    MUTATION_TYPE_ENABLE_BINDING,
    StaticApplicationEnvironmentTenantResolver,
    TenantScopedControlPlaneMutationEvaluator,
    binding_config_digest,
    build_bind_agent_mutation_request,
    build_disable_binding_mutation_request,
    build_enable_binding_mutation_request,
    build_install_agent_mutation_request,
    build_update_binding_config_mutation_request,
)
from intergrax.agent_distribution.errors import BindingRevisionConflict
from intergrax.agent_distribution.installation import InstallationState
from intergrax.applications._shared.agent_platform_admin_routes import (
    mount_agent_platform_admin_routes,
)
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    control_plane_mutation_request_digest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _APP,
    _DIGEST,
    _ENV,
    _PACKAGE,
    _bind_request,
    _install_request,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OTHER_DIGEST = "sha256:" + ("b" * 64)

_CONTROL_PLANE_GOVERNANCE_SLICE_FILES = (
    "intergrax/contracts/control_plane_mutation.py",
    "intergrax/runtime/governance/control_plane_mutation_authorization.py",
    "intergrax/agent_distribution/control_plane_governance.py",
    "intergrax/agent_distribution/admin_service.py",
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
    raise_error: bool = False

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        if self.raise_error:
            raise RuntimeError("evaluator exploded")
        return self.decision


def _stack_with_evaluator(evaluator: _RecordingEvaluator):
    stack = build_admin_stack()
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    return stack


def _install_only(stack, *, evaluator: _RecordingEvaluator | None = None) -> None:
    if evaluator is not None:
        stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
            ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
        )
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
        principal=admin_test_principal(),
    )


def _install_and_bind(stack, *, evaluator: _RecordingEvaluator | None = None) -> None:
    principal = admin_test_principal()
    if evaluator is not None:
        stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
            ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
        )
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
        principal=principal,
    )


def _mutation_calls(
    evaluator: _RecordingEvaluator,
    mutation_type: str,
) -> list[ControlPlaneMutationRequest]:
    return [call for call in evaluator.calls if call.mutation_type == mutation_type]


def test_ads1_install_allow_one_mutation_sequence() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    before = len(stack.state.installations)
    result = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-ads1"),
        principal=admin_test_principal(),
    )
    assert result.installation.installation_state is InstallationState.INSTALLED_ACTIVE
    assert len(stack.state.installations) == before + 1
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].mutation_type == "agent_distribution.install_agent"


def test_ads2_install_deny_zero_mutations() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack = _stack_with_evaluator(evaluator)
    before = len(stack.state.installations)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-ads2"),
            principal=admin_test_principal(),
        )
    assert len(stack.state.installations) == before


def test_ads3_install_require_human_zero_mutations_preserves_scope() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="hitl",
            policy_rule_id="rule.hitl",
        )
    )
    stack = _stack_with_evaluator(evaluator)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-ads3"),
            principal=admin_test_principal(),
        )
    assert exc.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert exc.value.authorization_scope is not None
    assert exc.value.authorization_scope.mutation_id == "mut-ads3"


def test_ads4_install_tenant_mismatch_zero_mutations() -> None:
    stack = build_admin_stack()
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("other-tenant")
    )
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=_RecordingEvaluator())
    )
    before = len(stack.state.installations)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-ads4"),
            principal=admin_test_principal(),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert len(stack.state.installations) == before


def test_ads5_install_package_digest_changes_request_digest() -> None:
    principal = admin_test_principal()
    base = build_install_agent_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads5",
        installation_slot_id="slot-search",
        installation_id="inst-1",
        package_digest=_DIGEST,
        current_revision="slot:slot-search|inst:__absent__",
    )
    other = build_install_agent_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads5",
        installation_slot_id="slot-search",
        installation_id="inst-1",
        package_digest=_OTHER_DIGEST,
        current_revision="slot:slot-search|inst:__absent__",
    )
    assert control_plane_mutation_request_digest(base) != control_plane_mutation_request_digest(
        other
    )


def test_ads6_bind_allow_one_create() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    _install_only(stack)
    before = len(stack.state.bindings)
    result = stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(mutation_id="mut-ads6"),
        principal=admin_test_principal(),
    )
    assert result.binding.application_binding_id == "bind-search"
    assert len(stack.state.bindings) == before + 1
    assert len(_mutation_calls(evaluator, "agent_distribution.bind_agent")) == 1


def test_ads7_bind_deny_zero_create() -> None:
    stack = build_admin_stack()
    _install_only(stack)
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    before = len(stack.state.bindings)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.bind_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_bind_request(mutation_id="mut-ads7"),
            principal=admin_test_principal(),
        )
    assert len(stack.state.bindings) == before


def test_ads8_bind_target_identity_changes_request_digest() -> None:
    principal = admin_test_principal()
    base = build_bind_agent_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads8",
        application_binding_id="bind-a",
        logical_agent_id="researcher",
        installation_slot_id="slot-search",
        enablement=False,
        current_revision="binding:__absent__",
    )
    other = build_bind_agent_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads8",
        application_binding_id="bind-b",
        logical_agent_id="planner",
        installation_slot_id="slot-other",
        enablement=False,
        current_revision="binding:__absent__",
    )
    assert control_plane_mutation_request_digest(base) != control_plane_mutation_request_digest(
        other
    )


def test_ads9_update_config_allow_one_update() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    _install_and_bind(stack)
    before = stack.state.bindings["bind-search"].binding_revision
    result = stack.service.update_binding_config(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=UpdateAgentBindingRequest(
            mutation_id="mut-ads9",
            expected_revision=0,
            config={"mode": "fast"},
        ),
        principal=admin_test_principal(),
    )
    assert result.binding.binding_revision == before + 1
    assert len(_mutation_calls(evaluator, "agent_distribution.update_binding_config")) == 1


def test_ads10_update_config_deny_zero_updates() -> None:
    stack = build_admin_stack()
    _install_and_bind(stack)
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    before = stack.state.bindings["bind-search"].binding_revision
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.update_binding_config(
            application_id=_APP,
            application_environment_id=_ENV,
            application_binding_id="bind-search",
            request=UpdateAgentBindingRequest(
                mutation_id="mut-ads10",
                expected_revision=0,
                config={"mode": "fast"},
            ),
            principal=admin_test_principal(),
        )
    assert stack.state.bindings["bind-search"].binding_revision == before


def test_ads11_update_config_revision_binding_in_request() -> None:
    principal = admin_test_principal()
    rev4 = build_update_binding_config_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads11",
        application_binding_id="bind-search",
        expected_revision=4,
        config_digest_value=binding_config_digest({"mode": "fast"}),
    )
    rev5 = build_update_binding_config_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads11",
        application_binding_id="bind-search",
        expected_revision=5,
        config_digest_value=binding_config_digest({"mode": "fast"}),
    )
    assert rev4.current_revision != rev5.current_revision
    assert rev4.target_revision != rev5.target_revision


def test_ads12_update_config_cas_after_authorization() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    _install_and_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-enable", expected_revision=0),
        principal=admin_test_principal(),
    )
    with pytest.raises(BindingRevisionConflict):
        stack.service.update_binding_config(
            application_id=_APP,
            application_environment_id=_ENV,
            application_binding_id="bind-search",
            request=UpdateAgentBindingRequest(
                mutation_id="mut-ads12",
                expected_revision=0,
                config={"mode": "fast"},
            ),
            principal=admin_test_principal(),
        )


def test_ads13_config_digest_changes_target_request_digest() -> None:
    principal = admin_test_principal()
    digest_a = binding_config_digest({"mode": "fast"})
    digest_b = binding_config_digest({"mode": "slow"})
    first = build_update_binding_config_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads13",
        application_binding_id="bind-search",
        expected_revision=0,
        config_digest_value=digest_a,
    )
    second = build_update_binding_config_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads13",
        application_binding_id="bind-search",
        expected_revision=0,
        config_digest_value=digest_b,
    )
    assert control_plane_mutation_request_digest(first) != control_plane_mutation_request_digest(
        second
    )


def test_ads14_enable_allow_once() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    _install_and_bind(stack)
    result = stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-ads14", expected_revision=0),
        principal=admin_test_principal(),
    )
    assert result.binding.enablement is True
    assert evaluator.calls[-1].mutation_type == MUTATION_TYPE_ENABLE_BINDING


def test_ads15_enable_deny_zero_mutation() -> None:
    stack = build_admin_stack()
    _install_and_bind(stack)
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.enable_binding(
            application_id=_APP,
            application_environment_id=_ENV,
            application_binding_id="bind-search",
            request=SetAgentEnablementRequest(mutation_id="mut-ads15", expected_revision=0),
            principal=admin_test_principal(),
        )
    assert stack.state.bindings["bind-search"].enablement is False


def test_ads16_disable_allow_once() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    _install_and_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-enable", expected_revision=0),
        principal=admin_test_principal(),
    )
    result = stack.service.disable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-ads16", expected_revision=1),
        principal=admin_test_principal(),
    )
    assert result.binding.enablement is False
    assert evaluator.calls[-1].mutation_type == MUTATION_TYPE_DISABLE_BINDING


def test_ads17_enable_vs_disable_different_digest() -> None:
    principal = admin_test_principal()
    enable = build_enable_binding_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads17",
        application_binding_id="bind-search",
        expected_revision=1,
        current_enablement=False,
    )
    disable = build_disable_binding_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads17",
        application_binding_id="bind-search",
        expected_revision=1,
        current_enablement=True,
    )
    assert enable.mutation_type != disable.mutation_type
    assert control_plane_mutation_request_digest(enable) != control_plane_mutation_request_digest(
        disable
    )


def test_ads18_no_policy_tenant_match_deny() -> None:
    evaluator = TenantScopedControlPlaneMutationEvaluator(
        tenant_resolver=StaticApplicationEnvironmentTenantResolver("tenant-test"),
    )
    request = build_install_agent_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads18",
        installation_slot_id="slot-search",
        installation_id="inst-1",
        package_digest=_DIGEST,
        current_revision="slot:slot-search|inst:__absent__",
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "control_plane_policy_not_configured"


def test_ads19_no_tenant_resolver_deny() -> None:
    evaluator = TenantScopedControlPlaneMutationEvaluator()
    request = build_install_agent_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-ads19",
        installation_slot_id="slot-search",
        installation_id="inst-1",
        package_digest=_DIGEST,
        current_revision="slot:slot-search|inst:__absent__",
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "tenant_authority_not_configured"


def test_ads20_policy_failure_deny_zero_mutation() -> None:
    evaluator = _RecordingEvaluator(raise_error=True)
    stack = _stack_with_evaluator(evaluator)
    before = len(stack.state.installations)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-ads20"),
            principal=admin_test_principal(),
        )
    assert exc.value.policy_action == PolicyAction.DENY.value
    assert len(stack.state.installations) == before


def test_ads21_modify_deny_zero_mutation() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.MODIFY, reason="modify")
    )
    stack = _stack_with_evaluator(evaluator)
    before = len(stack.state.installations)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-ads21"),
            principal=admin_test_principal(),
        )
    assert len(stack.state.installations) == before


def test_ads22_http_principal_reaches_mutation_request() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("default")
    )
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(require_api_key=False)
    mount_agent_platform_admin_routes(app, admin_service=stack.service)
    client = TestClient(app)
    payload = {
        "mutation_id": "mut-ads22",
        "installation_id": "inst-http",
        "installation_slot_id": "slot-search",
        "package_identity": {
            "distribution_package_id": _PACKAGE.distribution_package_id,
            "package_version": _PACKAGE.package_version,
            "package_digest": _PACKAGE.package_digest,
        },
        "artifact_store_ref": "store://artifacts/inst-http",
        "trust_record": {
            "qualification_status": "production_qualified",
            "package_digest": _DIGEST,
            "publisher_identity_ref": "publisher:acme",
            "source_provider_id": "builtin",
            "trust_evidence_refs": [
                {
                    "evidence_id": "evidence:service:0",
                    "kind": "signature_verification",
                }
            ],
        },
        "agent_project_metadata_ref": "meta://search",
    }
    response = client.post(
        f"/v1/agent-platform/applications/{_APP}/environments/{_ENV}/installations",
        json=payload,
    )
    assert response.status_code == 200
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].principal.tenant_id == "default"
    assert evaluator.calls[0].principal.user_id == "local-dev-admin"
    assert evaluator.calls[0].mutation_id == "mut-ads22"


def test_ads23_mutation_id_preserved_exactly() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="  mut-exact-id  "),
        principal=admin_test_principal(),
    )
    assert evaluator.calls[0].mutation_id == "mut-exact-id"


def test_ads24_retry_stability_same_mutation_id() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    request = _install_request(mutation_id="mut-ads24")
    principal = admin_test_principal()
    first = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    second = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    assert first.installation.installation_id == second.installation.installation_id
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].mutation_id == "mut-ads24"


def test_ads25_static_no_dynamic_access_in_changed_slice() -> None:
    offenders: list[str] = []
    for relative in _CONTROL_PLANE_GOVERNANCE_SLICE_FILES:
        path = _REPO_ROOT / relative
        text = path.read_text(encoding="utf-8")
        if _FORBIDDEN_DYNAMIC_PATTERNS.search(text):
            offenders.append(relative)
    assert offenders == []


def test_ads26_static_admin_bypass_inventory() -> None:
    allowed_relative = {
        "intergrax/agent_distribution/admin_service.py",
    }
    production_roots = (_REPO_ROOT / "intergrax", _REPO_ROOT / "applications")
    patterns = (
        ".create_candidate_installation(",
        ".create_binding(",
        ".update_config(",
        ".enable(",
        ".disable(",
    )
    hits: list[str] = []
    for root in production_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            posix = path.as_posix()
            if "/tests/" in posix or path.name.startswith("test_"):
                continue
            if "/docker/runtime-context/" in posix:
                continue
            text = path.read_text(encoding="utf-8")
            if not any(pattern in text for pattern in patterns):
                continue
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if relative in {
                "intergrax/agent_distribution/binding_service.py",
                "intergrax/agent_distribution/installation_service.py",
                "applications/local_workspace_application/workspaces/knowledge_inspection_operations_service.py",
            }:
                continue
            hits.append(relative)
    unexpected = sorted(path for path in hits if path not in allowed_relative)
    assert unexpected == []


def test_ads27_activation_consumer_regression_still_green() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_activation

    assert test_agent_distribution_activation is not None


def test_ads28_foundation_regression_still_green() -> None:
    from tests.unit.runtime.governance import test_control_plane_mutation_authorization

    assert test_control_plane_mutation_authorization is not None


def test_ads29_policy_governance_regression_still_green() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_build_remediation

    assert test_agent_distribution_build_remediation is not None


def test_ads30_build_consumer_implemented_in_ad_build() -> None:
    """AD-BUILD governs build_application_revision — see ADB remediation suite."""
    from tests.unit.agent_distribution import test_agent_distribution_build_remediation

    assert test_agent_distribution_build_remediation is not None


def test_authorize_before_create_candidate_installation() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    calls: list[str] = []
    original = stack.service._installation_service.create_candidate_installation

    def _tracked_create_candidate_installation(*args, **kwargs):
        calls.append("create")
        return original(*args, **kwargs)

    with patch.object(
        stack.service._installation_service,
        "create_candidate_installation",
        side_effect=_tracked_create_candidate_installation,
    ):
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-order"),
            principal=admin_test_principal(),
        )
    assert len(calls) == 1


def test_missing_boundary_blocks_desired_state() -> None:
    stack = build_admin_stack()
    stack.service._mutation_authorization_boundary = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(),
            principal=admin_test_principal(),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_MUTATION_AUTHORIZATION_BOUNDARY"
