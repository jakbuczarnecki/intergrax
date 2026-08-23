# © Artur Czarnecki. All rights reserved.

"""Agent Distribution control-plane mutation governance tests (AD1–AD18)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    AgentPlatformAdminGovernanceBlockedError,
    RollbackRuntimeRevisionRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.control_plane_governance import (
    build_activation_mutation_request,
    build_rollback_mutation_request,
    serving_revision_token,
)
from intergrax.agent_distribution.errors import RuntimeActivationConflict
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
    _build_request,
    _install_request,
    _rollback_request,
    build_admin_stack,
    admin_test_principal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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


class _StaticTenantResolver:
    def __init__(self, tenant_id: str) -> None:
        self._tenant_id = tenant_id

    def resolve_tenant_id(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> str:
        del application_id, application_environment_id
        return self._tenant_id


def _seed_validated_revision(stack, revision_id: str) -> object:
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
    )
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    return stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request(revision_id),
    )


def _stack_with_evaluator(evaluator: _RecordingEvaluator):
    stack = build_admin_stack()
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    return stack


def test_ad1_activation_allow_commits_once() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad1")
    before = stack.state.serving_records
    result = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-ad1",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-ad1",
        ),
    )
    assert result.traffic_serving_revision_id == "rev-ad1"
    assert result.authorization_evidence is not None
    assert result.authorization_evidence.mutation_id == "mut-ad1"
    assert len(evaluator.calls) == 1
    assert len(before) <= len(stack.state.serving_records)


def test_ad2_activation_deny_zero_commits() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad2")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-ad2",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-ad2",
            ),
        )
    assert exc.value.policy_action == PolicyAction.DENY.value
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_ad3_activation_require_human_zero_commits_preserves_scope() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs_approval",
            policy_rule_id="rule.hitl",
        )
    )
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad3")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-ad3",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-ad3",
            ),
        )
    assert exc.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert exc.value.authorization_scope is not None
    assert exc.value.authorization_scope.mutation_id == "mut-ad3"
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_ad4_rollback_allow_commits_once() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    first = _seed_validated_revision(stack, "rev-ad4-a")
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-ad4-a",
            artifact_locator=first.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-ad4-a",
        ),
    )
    second = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-ad4-b"),
    )
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-ad4-b",
            runtime_revision_id="rev-ad4-b",
            artifact_locator=second.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            expected_serving_pointer_revision=1,
            expected_prior_traffic_revision_id="rev-ad4-a",
        ),
    )
    evaluator.calls.clear()
    rolled = stack.service.rollback_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_rollback_request(
            expected_current_traffic_revision_id="rev-ad4-b",
            expected_serving_pointer_revision=2,
            mutation_id="mut-ad4-rollback",
        ),
    )
    assert rolled.restored_revision_id == "rev-ad4-a"
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].mutation_type == "agent_distribution.rollback_runtime_revision"


def test_ad5_rollback_deny_zero_commits() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    first = _seed_validated_revision(stack, "rev-ad5-a")
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-ad5-a",
            artifact_locator=first.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-ad5-a",
        ),
    )
    second = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-ad5-b"),
    )
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-ad5-b",
            runtime_revision_id="rev-ad5-b",
            artifact_locator=second.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            expected_serving_pointer_revision=1,
            expected_prior_traffic_revision_id="rev-ad5-a",
        ),
    )
    evaluator.decision = PolicyDecision(action=PolicyAction.DENY, reason="deny")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.rollback_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_rollback_request(
                expected_current_traffic_revision_id="rev-ad5-b",
                expected_serving_pointer_revision=2,
                mutation_id="mut-ad5-rollback",
            ),
        )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id == "rev-ad5-b"


def test_ad6_rollback_require_human_zero_commits() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    first = _seed_validated_revision(stack, "rev-ad6-a")
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-ad6-a",
            artifact_locator=first.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-ad6-a",
        ),
    )
    second = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-ad6-b"),
    )
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-ad6-b",
            runtime_revision_id="rev-ad6-b",
            artifact_locator=second.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            expected_serving_pointer_revision=1,
            expected_prior_traffic_revision_id="rev-ad6-a",
        ),
    )
    evaluator.decision = PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="hitl")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.rollback_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_rollback_request(
                expected_current_traffic_revision_id="rev-ad6-b",
                expected_serving_pointer_revision=2,
                mutation_id="mut-ad6-rollback",
            ),
        )
    assert exc.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id == "rev-ad6-b"


def test_ad7_wrong_tenant_authority_denies_without_mutation() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    stack.service._environment_tenant_resolver = _StaticTenantResolver("tenant-owned")  # type: ignore[attr-defined]
    built = _seed_validated_revision(stack, "rev-ad7")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-ad7",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-ad7",
            ),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert len(evaluator.calls) == 0


def test_ad8_stale_state_cas_rejects_after_allow() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad8")
    with pytest.raises(RuntimeActivationConflict):
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-ad8",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                expected_serving_pointer_revision=99,
                mutation_id="mut-ad8",
            ),
        )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_ad9_target_change_digest_differs() -> None:
    principal = admin_test_principal()
    first = build_activation_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-same",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-b",
    )
    second = build_activation_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-same",
        current_traffic_revision_id=None,
        current_serving_pointer_revision=0,
        target_runtime_revision_id="rev-c",
    )
    assert control_plane_mutation_request_digest(first) != control_plane_mutation_request_digest(
        second
    )


def test_ad10_mutation_id_stable_on_retry() -> None:
    evaluator = _RecordingEvaluator()
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad10")
    result = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-ad10",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            mutation_id="mut-stable",
        ),
    )
    assert result.authorization_evidence is not None
    assert result.authorization_evidence.mutation_id == "mut-stable"
    assert evaluator.calls[0].mutation_id == "mut-stable"
    request = build_activation_mutation_request(
        principal=admin_test_principal(),
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-stable",
        current_traffic_revision_id="rev-ad10",
        current_serving_pointer_revision=1,
        target_runtime_revision_id="rev-ad10",
    )
    boundary = stack.service._mutation_authorization_boundary  # type: ignore[attr-defined]
    assert boundary is not None
    second = boundary.authorize(request)
    assert second.evidence.mutation_id == "mut-stable"


def test_ad11_activation_and_rollback_digest_distinct() -> None:
    principal = admin_test_principal()
    activate = build_activation_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-11",
        current_traffic_revision_id="rev-a",
        current_serving_pointer_revision=1,
        target_runtime_revision_id="rev-b",
    )
    rollback = build_rollback_mutation_request(
        principal=principal,
        application_id=_APP,
        application_environment_id=_ENV,
        mutation_id="mut-11",
        current_traffic_revision_id="rev-a",
        current_serving_pointer_revision=1,
        target_runtime_revision_id="rev-b",
    )
    assert activate.mutation_type != rollback.mutation_type
    assert control_plane_mutation_request_digest(activate) != control_plane_mutation_request_digest(
        rollback
    )


def test_ad12_policy_failure_zero_mutations() -> None:
    evaluator = _RecordingEvaluator(raise_error=True)
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad12")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-ad12",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-ad12",
            ),
        )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_ad13_modify_zero_mutations() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.MODIFY, reason="modify")
    )
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad13")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-ad13",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-ad13",
            ),
        )
    assert exc.value.authorization_evidence.policy_action is PolicyAction.DENY
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_ad16_projection_order_prepare_candidate_before_authorization() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
    )
    stack = _stack_with_evaluator(evaluator)
    built = _seed_validated_revision(stack, "rev-ad16")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.activate_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_activate_request(
                runtime_revision_id="rev-ad16",
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=_ARTIFACT,
                mutation_id="mut-ad16",
            ),
        )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None
    assert serving.serving_pointer_revision == 0


def test_revision_token_binds_pointer() -> None:
    assert serving_revision_token(
        traffic_revision_id="rev-a",
        serving_pointer_revision=7,
    ) != serving_revision_token(
        traffic_revision_id="rev-c",
        serving_pointer_revision=8,
    )
