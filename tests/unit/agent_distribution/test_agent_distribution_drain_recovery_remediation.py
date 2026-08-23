# © Artur Czarnecki. All rights reserved.

"""Agent Distribution drain/recovery remediation tests (DR1–DR27)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminBlockedError,
    AgentPlatformAdminGovernanceBlockedError,
    ActivateRuntimeRevisionRequest,
    CompleteRevisionDrainRequest,
    HandlePostCutoverFailureRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.control_plane_governance import (
    MUTATION_TYPE_COMPLETE_DRAIN,
    MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE,
    MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION,
    StaticApplicationEnvironmentTenantResolver,
    TenantScopedControlPlaneMutationEvaluator,
    build_complete_drain_mutation_request,
    build_mark_post_cutover_failure_mutation_request,
    build_rollback_mutation_request,
    drain_policy_digest,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    DrainActionOnTimeout,
    DrainPolicy,
)
from intergrax.agent_distribution.errors import RuntimeActivationConflict, RuntimeDrainError
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
    _ARTIFACT,
    _ENV,
    _activate_request,
    _bind_request,
    _build_revision,
    _install_request,
    admin_test_principal,
    allow_mutation_boundary,
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
_SERVICE_PRINCIPAL = RequestIdentity(
    tenant_id="tenant-test",
    user_id="recovery-worker",
    principal_type=PrincipalType.SERVICE,
    auth_subject="recovery-worker",
)

_CONTROL_PLANE_GOVERNANCE_SLICE_FILES = (
    "intergrax/agent_distribution/control_plane_governance.py",
    "intergrax/agent_distribution/admin_models.py",
    "intergrax/agent_distribution/admin_service.py",
    "intergrax/agent_distribution/activation.py",
)

_FORBIDDEN_DYNAMIC_PATTERNS = re.compile(
    r"getattr\s*\(|setattr\s*\(|hasattr\s*\(|__dict__|eval\s*\(|exec\s*\("
)

_PRODUCTION_SCAN_PATHS = (
    _REPO_ROOT / "intergrax" / "agent_distribution",
    _REPO_ROOT / "intergrax" / "applications",
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


def _provision_stack():
    stack = build_admin_stack()
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        allow_mutation_boundary()
    )
    principal = admin_test_principal()
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
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-enable", expected_revision=0),
        principal=principal,
    )
    return stack


def _stack_with_evaluator(evaluator: _RecordingEvaluator):
    stack = _provision_stack()
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    return stack


def _enable_and_build(stack, revision_id: str) -> None:
    principal = admin_test_principal()
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
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-enable", expected_revision=0),
        principal=principal,
    )
    _build_revision(stack, revision_id)


def _activate_pair(stack, *, rev_a: str = "rev-1", rev_b: str = "rev-2") -> object:
    principal = admin_test_principal()
    prior_boundary = stack.service._mutation_authorization_boundary  # type: ignore[attr-defined]
    stack.service._mutation_authorization_boundary = allow_mutation_boundary()  # type: ignore[attr-defined]
    try:
        built_a = _build_revision(stack, rev_a)
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=principal,
            request=_activate_request(
                runtime_revision_id=rev_a,
                artifact_locator=built_a.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id=f"mut-activate-{rev_a}",
            ),
        )
        built_b = _build_revision(stack, rev_b)
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=principal,
            request=ActivateRuntimeRevisionRequest(
                mutation_id=f"mut-activate-{rev_b}",
                runtime_revision_id=rev_b,
                artifact_locator=built_b.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                expected_serving_pointer_revision=1,
                expected_prior_traffic_revision_id=rev_a,
            ),
        )
    finally:
        stack.service._mutation_authorization_boundary = prior_boundary  # type: ignore[attr-defined]
    prior = stack.service._deployment_instance_store.get_instance(  # type: ignore[attr-defined]
        _APP, _ENV, rev_a
    )
    assert prior is not None
    return prior


def _drain_request(
    *,
    runtime_revision_id: str,
    expected_record_revision: int,
    mutation_id: str = "mut-drain",
    policy: DrainPolicy | None = None,
) -> CompleteRevisionDrainRequest:
    return CompleteRevisionDrainRequest(
        mutation_id=mutation_id,
        runtime_revision_id=runtime_revision_id,
        expected_record_revision=expected_record_revision,
        policy=policy or DrainPolicy(timeout_seconds=30.0),
    )


def _complete_adapter_drain(stack, serving_unit_ref: str) -> None:
    adapter = stack.service._activation_service._deployment_adapter  # type: ignore[attr-defined]
    adapter.complete_drain(serving_unit_ref)


def test_dr1_complete_drain_allow_stops_once() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    prior = _activate_pair(stack)
    _complete_adapter_drain(stack, prior.serving_unit_ref or "")
    result = stack.service.complete_revision_drain(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_drain_request(
            runtime_revision_id="rev-1",
            expected_record_revision=prior.record_revision,
            mutation_id="mut-dr1",
        ),
    )
    assert result.instance_state is DeploymentInstanceState.STOPPED
    assert result.record_revision == prior.record_revision + 1
    instance = stack.service._deployment_instance_store.get_instance(  # type: ignore[attr-defined]
        _APP, _ENV, "rev-1"
    )
    assert instance is not None
    assert instance.instance_state is DeploymentInstanceState.STOPPED


def test_dr2_complete_drain_deny_zero_effects() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack = _stack_with_evaluator(evaluator)
    prior = _activate_pair(stack)
    _complete_adapter_drain(stack, prior.serving_unit_ref or "")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.complete_revision_drain(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_drain_request(
                runtime_revision_id="rev-1",
                expected_record_revision=prior.record_revision,
            ),
        )
    instance = stack.service._deployment_instance_store.get_instance(  # type: ignore[attr-defined]
        _APP, _ENV, "rev-1"
    )
    assert instance is not None
    assert instance.instance_state is DeploymentInstanceState.DRAINING
    assert instance.record_revision == prior.record_revision


def test_dr3_complete_drain_require_human_zero_effects() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="hitl")
    )
    stack = _stack_with_evaluator(evaluator)
    prior = _activate_pair(stack)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.complete_revision_drain(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_drain_request(
                runtime_revision_id="rev-1",
                expected_record_revision=prior.record_revision,
            ),
        )
    assert exc.value.authorization_evidence is not None
    assert exc.value.authorization_scope is not None
    instance = stack.service._deployment_instance_store.get_instance(  # type: ignore[attr-defined]
        _APP, _ENV, "rev-1"
    )
    assert instance is not None
    assert instance.instance_state is DeploymentInstanceState.DRAINING


def test_dr4_drain_tenant_mismatch_blocked() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    prior = _activate_pair(stack)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.complete_revision_drain(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=_OTHER_TENANT_PRINCIPAL,
            request=_drain_request(
                runtime_revision_id="rev-1",
                expected_record_revision=prior.record_revision,
            ),
        )


def test_dr5_drain_missing_policy_denies() -> None:
    evaluator = TenantScopedControlPlaneMutationEvaluator(
        tenant_resolver=StaticApplicationEnvironmentTenantResolver("tenant-test"),
    )
    request = build_complete_drain_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr5",
        runtime_revision_id="rev-1",
        record_revision=1,
        serving_unit_ref="unit-1",
        policy=DrainPolicy(timeout_seconds=5.0),
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "control_plane_policy_not_configured"


def test_dr6_drain_policy_failure_denies() -> None:
    evaluator = _RecordingEvaluator(raise_error=True)
    stack = _stack_with_evaluator(evaluator)
    prior = _activate_pair(stack)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.complete_revision_drain(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_drain_request(
                runtime_revision_id="rev-1",
                expected_record_revision=prior.record_revision,
            ),
        )


def test_dr7_drain_record_revision_binding() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    prior = _activate_pair(stack)
    with pytest.raises(RuntimeActivationConflict):
        stack.service.complete_revision_drain(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_drain_request(
                runtime_revision_id="rev-1",
                expected_record_revision=prior.record_revision + 1,
            ),
        )


def test_dr8_drain_policy_identity_changes_digest() -> None:
    policy_a = DrainPolicy(timeout_seconds=5.0)
    policy_b = DrainPolicy(
        timeout_seconds=5.0,
        action_on_timeout=DrainActionOnTimeout.MARK_FAILED,
    )
    request_a = build_complete_drain_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr8",
        runtime_revision_id="rev-1",
        record_revision=2,
        serving_unit_ref="unit-1",
        policy=policy_a,
    )
    request_b = build_complete_drain_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr8",
        runtime_revision_id="rev-1",
        record_revision=2,
        serving_unit_ref="unit-1",
        policy=policy_b,
    )
    assert drain_policy_digest(policy_a) != drain_policy_digest(policy_b)
    assert control_plane_mutation_request_digest(request_a) != control_plane_mutation_request_digest(
        request_b
    )


def test_dr9_drain_timeout_mark_failed_after_authorization() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    prior = _activate_pair(stack)
    adapter = stack.service._activation_service._deployment_adapter  # type: ignore[attr-defined]
    adapter.force_drain_timeout(prior.serving_unit_ref or "")
    with pytest.raises(RuntimeDrainError):
        stack.service.complete_revision_drain(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_drain_request(
                runtime_revision_id="rev-1",
                expected_record_revision=prior.record_revision,
                policy=DrainPolicy(
                    timeout_seconds=1.0,
                    action_on_timeout=DrainActionOnTimeout.MARK_FAILED,
                ),
            ),
        )
    assert len(evaluator.calls) == 1
    drain_calls = [
        call for call in evaluator.calls if call.mutation_type == MUTATION_TYPE_COMPLETE_DRAIN
    ]
    assert len(drain_calls) == 1
    instance = stack.service._deployment_instance_store.get_instance(  # type: ignore[attr-defined]
        _APP, _ENV, "rev-1"
    )
    assert instance is not None
    assert instance.instance_state is DeploymentInstanceState.FAILED


def test_dr10_no_dynamic_access_in_governance_slice() -> None:
    for relative in _CONTROL_PLANE_GOVERNANCE_SLICE_FILES:
        source = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        assert _FORBIDDEN_DYNAMIC_PATTERNS.search(source) is None, relative


def test_dr11_failure_mark_is_control_plane_mutation() -> None:
    request = build_mark_post_cutover_failure_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr11",
        runtime_revision_id="rev-n1",
        record_revision=2,
        current_instance_state="serving",
    )
    assert request.mutation_type == MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE


def test_dr12_failure_mark_wrong_tenant_denies() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    instance = stack.service._deployment_instance_store.get_instance(  # type: ignore[attr-defined]
        _APP, _ENV, "rev-n1"
    )
    assert instance is not None
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.handle_post_cutover_failure(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=_OTHER_TENANT_PRINCIPAL,
            request=HandlePostCutoverFailureRequest(
                mutation_id="mut-dr12",
                runtime_revision_id="rev-n1",
                failure_evidence_ref="health:failed",
                originating_activation_mutation_id="mut-activate-rev-n1",
                attempt_rollback=False,
            ),
        )


def test_dr13_recovery_rollback_requires_governance() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack = _stack_with_evaluator(evaluator)
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.handle_post_cutover_failure(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=HandlePostCutoverFailureRequest(
                mutation_id="mut-dr13-mark",
                recovery_mutation_id="mut-dr13-recovery",
                runtime_revision_id="rev-n1",
                failure_evidence_ref="health:failed",
                originating_activation_mutation_id="mut-activate-rev-n1",
            ),
        )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id == "rev-n1"


def test_dr14_recovery_current_deny_leaves_traffic() -> None:
    class _DenyRollbackEvaluator:
        def __init__(self) -> None:
            self.calls: list[ControlPlaneMutationRequest] = []

        def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
            self.calls.append(request)
            if request.mutation_type == MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE:
                return PolicyDecision(action=PolicyAction.ALLOW, reason="allow-mark")
            return PolicyDecision(action=PolicyAction.DENY, reason="deny-rollback")

    evaluator = _DenyRollbackEvaluator()
    stack = _stack_with_evaluator(evaluator)  # type: ignore[arg-type]
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.handle_post_cutover_failure(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=HandlePostCutoverFailureRequest(
                mutation_id="mut-dr14-mark",
                recovery_mutation_id="mut-dr14-recovery",
                runtime_revision_id="rev-n1",
                failure_evidence_ref="health:failed",
                originating_activation_mutation_id="mut-activate-rev-n1",
            ),
        )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id == "rev-n1"


def test_dr15_recovery_allow_rolls_back() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    result = stack.service.handle_post_cutover_failure(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=HandlePostCutoverFailureRequest(
            mutation_id="mut-dr15-mark",
            recovery_mutation_id="mut-dr15-recovery",
            runtime_revision_id="rev-n1",
            failure_evidence_ref="health:failed",
            originating_activation_mutation_id="mut-activate-rev-n1",
        ),
    )
    assert result.rollback_result is not None
    assert result.rollback_result.restored_revision_id == "rev-n"


def test_dr16_recovery_target_binding_changes_identity() -> None:
    first = build_rollback_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr16",
        current_traffic_revision_id="rev-b",
        current_serving_pointer_revision=2,
        target_runtime_revision_id="rev-a",
    )
    second = build_rollback_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr16",
        current_traffic_revision_id="rev-b",
        current_serving_pointer_revision=2,
        target_runtime_revision_id="rev-z",
    )
    assert control_plane_mutation_request_digest(first) != control_plane_mutation_request_digest(
        second
    )


def test_dr17_recovery_pointer_binding_rejects_stale_pointer() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    with pytest.raises(RuntimeActivationConflict):
        stack.service.rollback_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=__import__(
                "intergrax.agent_distribution.admin_models",
                fromlist=["RollbackRuntimeRevisionRequest"],
            ).RollbackRuntimeRevisionRequest(
                mutation_id="mut-dr17",
                expected_current_traffic_revision_id="rev-n1",
                expected_serving_pointer_revision=serving.serving_pointer_revision - 1,
            ),
        )


def test_dr18_failure_evidence_ref_not_authority() -> None:
    first = build_mark_post_cutover_failure_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr18",
        runtime_revision_id="rev-n1",
        record_revision=2,
        current_instance_state="serving",
    )
    second = build_mark_post_cutover_failure_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr18",
        runtime_revision_id="rev-n1",
        record_revision=2,
        current_instance_state="serving",
    )
    assert control_plane_mutation_request_digest(first) == control_plane_mutation_request_digest(
        second
    )


def test_dr19_service_principal_for_automated_recovery() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    result = stack.service.handle_post_cutover_failure(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=_SERVICE_PRINCIPAL,
        request=HandlePostCutoverFailureRequest(
            mutation_id="mut-dr19-mark",
            recovery_mutation_id="mut-dr19-recovery",
            runtime_revision_id="rev-n1",
            failure_evidence_ref="health:failed",
            originating_activation_mutation_id="mut-activate-rev-n1",
        ),
    )
    assert result.rollback_result is not None
    assert result.rollback_result.authorization_evidence is not None
    assert result.rollback_result.authorization_evidence.principal_type == PrincipalType.SERVICE


def test_dr20_tenant_resolver_required_for_automated_path() -> None:
    stack = build_admin_stack()
    stack.service._environment_tenant_resolver = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError):
        stack.service.handle_post_cutover_failure(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=_SERVICE_PRINCIPAL,
            request=HandlePostCutoverFailureRequest(
                mutation_id="mut-dr20",
                runtime_revision_id="rev-1",
                failure_evidence_ref="health:failed",
                originating_activation_mutation_id="mut-activate",
                attempt_rollback=False,
            ),
        )


def test_dr21_recovery_mutation_id_stable() -> None:
    stack = _stack_with_evaluator(_RecordingEvaluator())
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    result = stack.service.handle_post_cutover_failure(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=HandlePostCutoverFailureRequest(
            mutation_id="mut-dr21-mark",
            recovery_mutation_id="mut-dr21-recovery",
            runtime_revision_id="rev-n1",
            failure_evidence_ref="health:failed",
            originating_activation_mutation_id="mut-activate-rev-n1",
        ),
    )
    assert result.rollback_result is not None
    assert result.rollback_result.authorization_evidence is not None
    assert result.rollback_result.authorization_evidence.mutation_id == "mut-dr21-recovery"


def test_dr22_no_double_governance_on_recovery() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    _activate_pair(stack, rev_a="rev-n", rev_b="rev-n1")
    stack.service.handle_post_cutover_failure(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=HandlePostCutoverFailureRequest(
            mutation_id="mut-dr22-mark",
            recovery_mutation_id="mut-dr22-recovery",
            runtime_revision_id="rev-n1",
            failure_evidence_ref="health:failed",
            originating_activation_mutation_id="mut-activate-rev-n1",
        ),
    )
    rollback_calls = [
        call for call in evaluator.calls if call.mutation_type == MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION
    ]
    assert len(rollback_calls) == 1


def test_dr23_production_bypass_inventory() -> None:
    offenders: list[str] = []
    patterns = (
        re.compile(r"\.complete_drain\("),
        re.compile(r"\.mark_post_cutover_failure\("),
    )
    allowed_suffixes = (
        "activation.py",
        "deployment.py",
        "admin_service.py",
        "test_agent_distribution_activation.py",
        "test_agent_distribution_rollback.py",
        "test_agent_distribution_drain_recovery_remediation.py",
    )
    for root in _PRODUCTION_SCAN_PATHS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name in allowed_suffixes:
                continue
            if "test_" in path.name:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                if pattern.search(text):
                    offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == []


def test_dr24_activation_regression_smoke() -> None:
    stack = _provision_stack()
    built = _build_revision(stack, "rev-dr24")
    result = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-dr24",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-dr24",
        ),
    )
    assert result.traffic_serving_revision_id == "rev-dr24"


def test_dr25_desired_state_regression_smoke() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    installed = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-dr25-install"),
        principal=principal,
    )
    assert installed.installation.installation_id == "inst-1"


def test_dr26_build_regression_smoke() -> None:
    stack = build_admin_stack()
    _enable_and_build(stack, "rev-dr26")
    revision = stack.service.inspect_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-dr26",
    )
    assert revision.revision_state.value == "validated"


def test_dr27_foundation_tenant_scope_regression() -> None:
    evaluator = TenantScopedControlPlaneMutationEvaluator(
        tenant_resolver=StaticApplicationEnvironmentTenantResolver("tenant-test"),
    )
    request = build_complete_drain_mutation_request(
        principal=_OTHER_TENANT_PRINCIPAL,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-dr27",
        runtime_revision_id="rev-1",
        record_revision=1,
        serving_unit_ref="unit-1",
        policy=DrainPolicy(timeout_seconds=5.0),
    )
    decision = evaluator.evaluate(request)
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "tenant_authority_mismatch"
