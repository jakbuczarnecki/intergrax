# © Artur Czarnecki. All rights reserved.

"""Agent Distribution activation remediation tests (ADR1–ADR18)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminBlockedError,
    AgentPlatformAdminGovernanceBlockedError,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
    TenantScopedControlPlaneMutationEvaluator,
    build_activation_mutation_request,
)
from intergrax.agent_distribution.errors import RuntimeActivationConflict
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    build_reference_production_control_plane_governance,
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleGovernanceBlockedError,
    ReferenceProductionLifecycleLauncher,
)
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
from research_application.host.reference_lifecycle_input import build_research_reference_lifecycle_input
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _APP,
    _ARTIFACT,
    _ENV,
    _activate_request,
    admin_test_principal,
    allow_mutation_boundary,
    build_admin_stack,
)
from tests.unit.agent_distribution.test_agent_distribution_control_plane_governance import (
    _RecordingEvaluator,
    _seed_validated_revision,
    _stack_with_evaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CONTROL_PLANE_GOVERNANCE_SLICE_FILES = (
    "intergrax/contracts/control_plane_mutation.py",
    "intergrax/runtime/governance/control_plane_mutation_authorization.py",
    "intergrax/agent_distribution/control_plane_governance.py",
    "intergrax/agent_distribution/admin_service.py",
    "intergrax/applications/_shared/reference_production_lifecycle.py",
    "intergrax/applications/_shared/reference_production_governance_wiring.py",
    "intergrax/applications/_shared/agent_platform_admin_routes.py",
)

_FORBIDDEN_DYNAMIC_PATTERNS = re.compile(
    r"getattr\s*\(|setattr\s*\(|hasattr\s*\(|__dict__|eval\s*\(|exec\s*\("
)


@dataclass
class _AllowEvaluator:
    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        del request
        return PolicyDecision(action=PolicyAction.ALLOW, reason="allow")


def test_adr1_no_authority_config_denies_without_commit() -> None:
    evaluator = TenantScopedControlPlaneMutationEvaluator()
    request = build_activation_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-adr1",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-adr1",
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "tenant_authority_not_configured"


def test_adr2_tenant_match_without_policy_denies() -> None:
    evaluator = TenantScopedControlPlaneMutationEvaluator(
        tenant_resolver=StaticApplicationEnvironmentTenantResolver("tenant-test"),
    )
    request = build_activation_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-adr2",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-adr2",
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "control_plane_policy_not_configured"


def test_adr3_tenant_mismatch_denies_even_when_inner_would_allow() -> None:
    inner = _AllowEvaluator()
    evaluator = TenantScopedControlPlaneMutationEvaluator(
        tenant_resolver=StaticApplicationEnvironmentTenantResolver("tenant-owned"),
        inner=inner,
    )
    request = build_activation_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-adr3",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-adr3",
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "tenant_authority_mismatch"


def test_adr4_tenant_match_and_policy_allow_commits_once() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    built = _seed_validated_revision(stack, "rev-adr4")
    result = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-adr4",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-adr4",
        ),
    )
    assert result.traffic_serving_revision_id == "rev-adr4"


def test_adr5_policy_deny_zero_commits() -> None:
    stack = _stack_with_evaluator(
        _RecordingEvaluator(decision=PolicyDecision(action=PolicyAction.DENY, reason="deny"))
    )
    built = _seed_validated_revision(stack, "rev-adr5")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-adr5",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-adr5",
            ),
        )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_adr6_require_human_zero_commits_preserves_scope() -> None:
    stack = _stack_with_evaluator(
        _RecordingEvaluator(
            decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="hitl")
        )
    )
    built = _seed_validated_revision(stack, "rev-adr6")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-adr6",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-adr6",
            ),
        )
    assert exc.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert exc.value.authorization_scope is not None
    assert exc.value.authorization_scope.mutation_id == "mut-adr6"


def test_adr7_reference_production_allow_commits() -> None:
    composition = create_reference_production_process_composition()
    settings = ResearchBackendSettings(use_nexus_loop=True)
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-adr7",
    )
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    result = launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
    )
    assert result.runtime_revision_id == "rev-adr7"
    assert result.serving_pointer_revision == 1


def test_adr8_reference_production_deny_zero_commit() -> None:
    composition = create_reference_production_process_composition()
    settings = ResearchBackendSettings(use_nexus_loop=True)
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-adr8",
    )
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    governance = build_reference_production_control_plane_governance(env)
    deny_evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    launcher = ReferenceProductionLifecycleLauncher(
        composition,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=deny_evaluator
        ),
        environment_tenant_resolver=governance.environment_tenant_resolver,
    )
    with pytest.raises(ReferenceProductionLifecycleGovernanceBlockedError):
        launcher.deploy_and_activate(
            projection_input,
            activation_request,
            principal=governance.principal,
        )
    stores = composition.agent_platform_runtime.stores
    serving = stores.serving_store.get_serving_record(
        manifest.app_id,
        env.profile_id,
    )
    assert serving is None or serving.traffic_serving_revision_id is None


def test_adr9_reference_production_missing_boundary_fails_closed() -> None:
    composition = create_reference_production_process_composition()
    settings = ResearchBackendSettings(use_nexus_loop=True)
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-adr9",
    )
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    governance = build_reference_production_control_plane_governance(env)
    launcher = ReferenceProductionLifecycleLauncher(
        composition,
        environment_tenant_resolver=governance.environment_tenant_resolver,
    )
    with pytest.raises(ReferenceProductionLifecycleGovernanceBlockedError):
        launcher.deploy_and_activate(
            projection_input,
            activation_request,
            principal=governance.principal,
        )


def test_adr10_reference_production_wrong_tenant_zero_commit() -> None:
    composition = create_reference_production_process_composition()
    settings = ResearchBackendSettings(use_nexus_loop=True)
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-adr10",
    )
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    wrong_principal = RequestIdentity(
        tenant_id="wrong-tenant",
        user_id="svc",
        principal_type=PrincipalType.SERVICE,
        auth_subject="svc",
    )
    with pytest.raises(ReferenceProductionLifecycleGovernanceBlockedError):
        launcher.deploy_and_activate(
            projection_input,
            activation_request,
            principal=wrong_principal,
        )


def test_adr11_reference_production_require_human_zero_commit() -> None:
    composition = create_reference_production_process_composition()
    settings = ResearchBackendSettings(use_nexus_loop=True)
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-adr11",
    )
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    governance = build_reference_production_control_plane_governance(env)
    launcher = ReferenceProductionLifecycleLauncher(
        composition,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(
                decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="hitl")
            )
        ),
        environment_tenant_resolver=governance.environment_tenant_resolver,
    )
    with pytest.raises(ReferenceProductionLifecycleGovernanceBlockedError) as exc:
        launcher.deploy_and_activate(
            projection_input,
            activation_request,
            principal=governance.principal,
        )
    assert exc.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert exc.value.authorization_scope is not None


def test_adr12_reference_path_reuses_activation_request_mutation_id() -> None:
    composition = create_reference_production_process_composition()
    settings = ResearchBackendSettings(use_nexus_loop=True)
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-adr12",
    )
    activation_request = activation_request.model_copy(update={"mutation_id": "mut-adr12-exact"})
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    governance = build_reference_production_control_plane_governance(env)
    recording = _RecordingEvaluator()
    launcher = ReferenceProductionLifecycleLauncher(
        composition,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=TenantScopedControlPlaneMutationEvaluator(
                tenant_resolver=governance.environment_tenant_resolver,
                inner=recording,
            )
        ),
        environment_tenant_resolver=governance.environment_tenant_resolver,
    )
    launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
    )
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_id == "mut-adr12-exact"


def test_adr13_revision_token_parity_between_admin_and_reference() -> None:
    principal = admin_test_principal()
    admin_request = build_activation_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-parity",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-parity",
    )
    reference_request = build_activation_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-parity-other",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-parity",
    )
    for field in (
        "resource_type",
        "resource_id",
        "resource_scope",
        "current_revision",
        "target_revision",
        "mutation_type",
    ):
        assert getattr(admin_request, field) == getattr(reference_request, field)


def test_adr14_cas_regression_after_allow() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    built = _seed_validated_revision(stack, "rev-adr14")
    with pytest.raises(RuntimeActivationConflict):
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-adr14",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                expected_serving_pointer_revision=99,
                mutation_id="mut-adr14",
            ),
        )


def test_adr15_admin_missing_tenant_resolver_blocks_activation() -> None:
    stack = build_admin_stack()
    stack.service._environment_tenant_resolver = None  # type: ignore[attr-defined]
    built = _seed_validated_revision(stack, "rev-adr15")
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-adr15",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-adr15",
            ),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_TENANT_AUTHORITY"


def test_adr16_static_production_bypass_inventory() -> None:
    allowed_relative = {
        "intergrax/agent_distribution/admin_service.py",
        "intergrax/applications/_shared/reference_production_lifecycle.py",
        "intergrax/agent_distribution/activation.py",
    }
    production_roots = (_REPO_ROOT / "intergrax", _REPO_ROOT / "applications")
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
            if ".commit_activation(" not in text:
                continue
            relative = path.relative_to(_REPO_ROOT).as_posix()
            hits.append(relative)
    unexpected = sorted(path for path in hits if path not in allowed_relative)
    assert unexpected == []
    assert "intergrax/applications/_shared/reference_production_lifecycle.py" in hits


def test_adr17_foundation_boundary_still_fail_closed_on_missing_principal() -> None:
    boundary = allow_mutation_boundary()
    request = build_activation_mutation_request(
        principal=RequestIdentity(
            tenant_id="tenant-test",
            user_id="",
            principal_type=PrincipalType.USER,
            auth_subject="",
        ),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-adr17",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-adr17",
    )
    result = boundary.authorize(request)
    assert result.permitted is False


def test_r2_control_plane_slice_zero_dynamic_access() -> None:
    offenders: list[str] = []
    for relative in _CONTROL_PLANE_GOVERNANCE_SLICE_FILES:
        path = _REPO_ROOT / relative
        text = path.read_text(encoding="utf-8")
        if _FORBIDDEN_DYNAMIC_PATTERNS.search(text):
            offenders.append(relative)
    assert offenders == []


def test_r2_tenant_scoped_evaluator_delegates_via_typed_protocol() -> None:
    inner = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    )
    evaluator = TenantScopedControlPlaneMutationEvaluator(
        tenant_resolver=StaticApplicationEnvironmentTenantResolver("tenant-test"),
        inner=inner,
    )
    request = build_activation_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-r2-delegate",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-r2-delegate",
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.ALLOW
    assert len(inner.calls) == 1
    assert inner.calls[0] is request


def test_r2_evaluator_failure_preserves_real_evidence() -> None:
    request = build_activation_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-r2-evidence",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-r2-evidence",
    )
    expected_digest = control_plane_mutation_request_digest(request)
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=_RecordingEvaluator(raise_error=True)
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert "evaluator_failure" in result.decision.reason
    assert result.evidence.request_digest == expected_digest
    assert result.evidence.mutation_id == "mut-r2-evidence"
    assert result.evidence.mutation_type == request.mutation_type
    assert result.evidence.resource_scope == request.resource_scope
    assert result.evidence.resource_type == request.resource_type
    assert result.evidence.resource_id == request.resource_id
    assert result.evidence.current_revision == request.current_revision
    assert result.evidence.target_revision == request.target_revision
    assert result.evidence.mutation_id != "unknown"
    assert "0000000000000000" not in result.evidence.request_digest


def test_adr18_previous_ad_allow_path_still_commits() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    built = _seed_validated_revision(stack, "rev-adr18")
    result = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-adr18",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-adr18",
        ),
    )
    assert result.traffic_serving_revision_id == "rev-adr18"
