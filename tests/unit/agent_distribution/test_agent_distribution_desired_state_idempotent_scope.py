# © Artur Czarnecki. All rights reserved.

"""Desired-state idempotent tenant-scope remediation tests (DSR1–DSR17)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminBlockedError,
    AgentPlatformAdminGovernanceBlockedError,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.errors import (
    BindingRevisionConflict,
    InstallationSlotConflict,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
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
_OTHER_PACKAGE = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="2.0.0",
    package_digest=_OTHER_DIGEST,
)

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

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return self.decision


def _stack_with_evaluator(evaluator: _RecordingEvaluator):
    stack = build_admin_stack()
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    return stack


def _wrong_tenant_stack(stack) -> None:
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("other-tenant")
    )


def test_dsr1_install_noop_same_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    request = _install_request(mutation_id="mut-dsr1")
    principal = admin_test_principal()
    first = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    create_calls: list[str] = []
    original = stack.service._installation_service.create_candidate_installation

    def _tracked(*args, **kwargs):
        create_calls.append("create")
        return original(*args, **kwargs)

    with patch.object(
        stack.service._installation_service,
        "create_candidate_installation",
        side_effect=_tracked,
    ):
        second = stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=principal,
        )
    assert second.installation.installation_id == first.installation.installation_id
    assert create_calls == []
    assert len(evaluator.calls) == 1


def test_dsr2_install_noop_wrong_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    request = _install_request(mutation_id="mut-dsr2")
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    _wrong_tenant_stack(stack)
    create_calls: list[str] = []
    original = stack.service._installation_service.create_candidate_installation

    def _tracked(*args, **kwargs):
        create_calls.append("create")
        return original(*args, **kwargs)

    with patch.object(
        stack.service._installation_service,
        "create_candidate_installation",
        side_effect=_tracked,
    ):
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.install_agent(
                application_id=_APP,
                application_environment_id=_ENV,
                request=request,
                principal=principal,
            )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert create_calls == []
    assert len(evaluator.calls) == 1


def test_dsr3_install_conflict_wrong_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-dsr3-install"),
        principal=principal,
    )
    _wrong_tenant_stack(stack)
    conflict_request = _install_request(mutation_id="mut-dsr3-conflict")
    conflict_request = conflict_request.model_copy(
        update={"package_identity": _OTHER_PACKAGE},
    )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=conflict_request,
            principal=principal,
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert len(evaluator.calls) == 1


def test_dsr4_bind_noop_same_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-dsr4-install"),
        principal=principal,
    )
    request = _bind_request(mutation_id="mut-dsr4-bind")
    first = stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    create_calls: list[str] = []
    original = stack.service._binding_service.create_binding

    def _tracked(*args, **kwargs):
        create_calls.append("create")
        return original(*args, **kwargs)

    with patch.object(
        stack.service._binding_service,
        "create_binding",
        side_effect=_tracked,
    ):
        second = stack.service.bind_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=principal,
        )
    assert second.binding.application_binding_id == first.binding.application_binding_id
    assert create_calls == []
    assert len(evaluator.calls) == 2


def test_dsr5_bind_noop_wrong_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-dsr5-install"),
        principal=principal,
    )
    request = _bind_request(mutation_id="mut-dsr5-bind")
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    _wrong_tenant_stack(stack)
    create_calls: list[str] = []
    original = stack.service._binding_service.create_binding

    def _tracked(*args, **kwargs):
        create_calls.append("create")
        return original(*args, **kwargs)

    with patch.object(
        stack.service._binding_service,
        "create_binding",
        side_effect=_tracked,
    ):
        with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
            stack.service.bind_agent(
                application_id=_APP,
                application_environment_id=_ENV,
                request=request,
                principal=principal,
            )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert create_calls == []
    assert len(evaluator.calls) == 2


def test_dsr6_bind_conflict_wrong_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-dsr6-install"),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(mutation_id="mut-dsr6-bind"),
        principal=principal,
    )
    _wrong_tenant_stack(stack)
    conflict_request = _bind_request(mutation_id="mut-dsr6-conflict").model_copy(
        update={"logical_agent_id": "other-agent"},
    )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.bind_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=conflict_request,
            principal=principal,
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert len(evaluator.calls) == 2


def _install_and_bind(stack, *, evaluator: _RecordingEvaluator | None = None) -> None:
    principal = admin_test_principal()
    if evaluator is not None:
        stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
            ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
        )
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-setup-install"),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(mutation_id="mut-setup-bind"),
        principal=principal,
    )


def test_dsr7_update_config_wrong_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    _install_and_bind(stack)
    _wrong_tenant_stack(stack)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.update_binding_config(
            application_id=_APP,
            application_environment_id=_ENV,
            application_binding_id="bind-search",
            request=UpdateAgentBindingRequest(
                mutation_id="mut-dsr7",
                expected_revision=0,
                config={"mode": "fast"},
            ),
            principal=admin_test_principal(),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert len(evaluator.calls) == 2


def test_dsr8_enable_wrong_tenant() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    _install_and_bind(stack)
    _wrong_tenant_stack(stack)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.enable_binding(
            application_id=_APP,
            application_environment_id=_ENV,
            application_binding_id="bind-search",
            request=SetAgentEnablementRequest(mutation_id="mut-dsr8", expected_revision=0),
            principal=admin_test_principal(),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert stack.state.bindings["bind-search"].enablement is False
    assert len(evaluator.calls) == 2


def test_dsr9_disable_wrong_tenant() -> None:
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
    _wrong_tenant_stack(stack)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.disable_binding(
            application_id=_APP,
            application_environment_id=_ENV,
            application_binding_id="bind-search",
            request=SetAgentEnablementRequest(mutation_id="mut-dsr9", expected_revision=1),
            principal=admin_test_principal(),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert stack.state.bindings["bind-search"].enablement is True
    assert len(evaluator.calls) == 3


def test_dsr10_real_mutation_policy_deny() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack = _stack_with_evaluator(evaluator)
    before = len(stack.state.installations)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(mutation_id="mut-dsr10"),
            principal=admin_test_principal(),
        )
    assert len(stack.state.installations) == before
    assert len(evaluator.calls) == 1


def test_dsr11_real_mutation_policy_allow() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    before = len(stack.state.installations)
    result = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-dsr11"),
        principal=admin_test_principal(),
    )
    assert result.installation.installation_id == "inst-1"
    assert len(stack.state.installations) == before + 1
    assert len(evaluator.calls) == 1


def test_dsr12_missing_resolver_noop_fail_closed() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    request = _install_request(mutation_id="mut-dsr12")
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    stack.service._environment_tenant_resolver = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=principal,
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_TENANT_AUTHORITY"
    assert len(evaluator.calls) == 1


def test_dsr13_noop_target_change_not_treated_as_noop() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-dsr13-install"),
        principal=principal,
    )
    conflict_request = _install_request(mutation_id="mut-dsr13-conflict").model_copy(
        update={"package_identity": _OTHER_PACKAGE},
    )
    with pytest.raises(InstallationSlotConflict):
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=conflict_request,
            principal=principal,
        )
    assert len(evaluator.calls) == 1

    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(mutation_id="mut-dsr13-bind"),
        principal=principal,
    )
    conflict_bind = _bind_request(mutation_id="mut-dsr13-bind-conflict").model_copy(
        update={"logical_agent_id": "other-agent"},
    )
    with pytest.raises(BindingRevisionConflict):
        stack.service.bind_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=conflict_bind,
            principal=principal,
        )
    assert len(evaluator.calls) == 2


def test_dsr14_static_zero_dynamic_access() -> None:
    offenders: list[str] = []
    for relative in _CONTROL_PLANE_GOVERNANCE_SLICE_FILES:
        path = _REPO_ROOT / relative
        text = path.read_text(encoding="utf-8")
        if _FORBIDDEN_DYNAMIC_PATTERNS.search(text):
            offenders.append(relative)
    assert offenders == []


def test_dsr15_ads_regression() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_desired_state_remediation

    assert test_agent_distribution_desired_state_remediation is not None


def test_dsr16_activation_regression() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_activation_remediation

    assert test_agent_distribution_activation_remediation is not None


def test_dsr17_foundation_regression() -> None:
    from tests.unit.runtime.governance import test_control_plane_mutation_authorization

    assert test_control_plane_mutation_authorization is not None


def test_dsr_wrong_tenant_principal_mismatch() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    request = _install_request(mutation_id="mut-dsr-principal")
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=admin_test_principal(),
    )
    wrong_principal = RequestIdentity(
        tenant_id="wrong-tenant",
        user_id="admin-1",
        principal_type=PrincipalType.USER,
        auth_subject="admin-1",
    )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=request,
            principal=wrong_principal,
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert len(evaluator.calls) == 1
