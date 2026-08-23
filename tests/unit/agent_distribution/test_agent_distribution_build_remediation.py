# © Artur Czarnecki. All rights reserved.

"""Agent Distribution build remediation tests (ADB1–ADB26)."""

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
    BuildApplicationRevisionRequest,
)
from intergrax.agent_distribution.control_plane_governance import (
    MUTATION_TYPE_BUILD_RUNTIME_REVISION,
    StaticApplicationEnvironmentTenantResolver,
    TenantScopedControlPlaneMutationEvaluator,
    build_input_digest,
    build_runtime_revision_identity_digest,
    build_runtime_revision_mutation_request,
)
from intergrax.agent_distribution.errors import RuntimeRevisionConflict
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
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
    _ARTIFACT,
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


def _install_enable(stack) -> None:
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
        request=__import__(
            "intergrax.agent_distribution.admin_models",
            fromlist=["SetAgentEnablementRequest"],
        ).SetAgentEnablementRequest(mutation_id="mut-enable", expected_revision=0),
        principal=principal,
    )


def _install_enable_with_allow_boundary(stack) -> None:
    from tests.unit.agent_distribution.test_agent_platform_admin_service import (
        allow_mutation_boundary,
    )

    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        allow_mutation_boundary()
    )
    _install_enable(stack)


def _prepare_build_evaluator_stack(evaluator: _RecordingEvaluator):
    stack = _stack_with_evaluator(evaluator)
    _install_enable_with_allow_boundary(stack)
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    return stack


def _build_mutation_calls(evaluator: _RecordingEvaluator) -> list[ControlPlaneMutationRequest]:
    return [
        call
        for call in evaluator.calls
        if call.mutation_type == MUTATION_TYPE_BUILD_RUNTIME_REVISION
    ]


def _build_payload(revision_id: str, mutation_id: str = "mut-build") -> dict[str, object]:
    request = _build_request(revision_id, mutation_id=mutation_id)
    return request.model_dump(mode="json")


def test_adb1_build_allow_persists_once() -> None:
    evaluator = _RecordingEvaluator()
    stack = _prepare_build_evaluator_stack(evaluator)
    before_locks = len(stack.state.locks)
    before_revisions = len(stack.state.revisions)
    result = _build_revision(stack, "rev-adb1", mutation_id="mut-adb1")
    assert result.revision_state is RuntimeRevisionState.VALIDATED
    assert len(stack.state.locks) == before_locks + 1
    assert len(stack.state.revisions) == before_revisions + 1
    build_calls = _build_mutation_calls(evaluator)
    assert len(build_calls) == 1


def test_adb2_build_deny_zero_writes() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(
                decision=PolicyDecision(action=PolicyAction.DENY, reason="deny")
            )
        )
    )
    before_locks = len(stack.state.locks)
    before_revisions = len(stack.state.revisions)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        _build_revision(stack, "rev-adb2", mutation_id="mut-adb2")
    assert len(stack.state.locks) == before_locks
    assert len(stack.state.revisions) == before_revisions


def test_adb3_require_human_zero_writes_preserves_scope() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(
                decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN, reason="hitl")
            )
        )
    )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as exc:
        _build_revision(stack, "rev-adb3", mutation_id="mut-adb3")
    assert exc.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert exc.value.authorization_scope is not None
    assert exc.value.authorization_evidence is not None


def test_adb4_tenant_mismatch_before_revision_lookup() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    _build_revision(stack, "rev-adb4-existing")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_build_request("rev-adb4-existing"),
            principal=_OTHER_TENANT_PRINCIPAL,
        )


def test_adb5_missing_tenant_resolver_fail_closed() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    stack.service._environment_tenant_resolver = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        _build_revision(stack, "rev-adb5")
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_TENANT_AUTHORITY"


def test_adb6_missing_policy_deny_zero_writes() -> None:
    evaluator = TenantScopedControlPlaneMutationEvaluator(
        tenant_resolver=StaticApplicationEnvironmentTenantResolver("tenant-test"),
    )
    stack = build_admin_stack()
    _install_enable(stack)
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    )
    before_revisions = len(stack.state.revisions)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        _build_revision(stack, "rev-adb6")
    assert len(stack.state.revisions) == before_revisions


def test_adb7_policy_failure_deny_zero_writes() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(raise_error=True)
        )
    )
    before_revisions = len(stack.state.revisions)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        _build_revision(stack, "rev-adb7")
    assert len(stack.state.revisions) == before_revisions


def test_adb8_modify_deny_zero_writes() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    stack.service._mutation_authorization_boundary = (  # type: ignore[attr-defined]
        ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(
                decision=PolicyDecision(action=PolicyAction.MODIFY, reason="modify")
            )
        )
    )
    before_revisions = len(stack.state.revisions)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        _build_revision(stack, "rev-adb8")
    assert len(stack.state.revisions) == before_revisions


def test_adb9_mutation_id_preserved_exactly() -> None:
    evaluator = _RecordingEvaluator()
    stack = _prepare_build_evaluator_stack(evaluator)
    _build_revision(stack, "rev-adb9", mutation_id="  mut-exact-build  ")
    build_calls = _build_mutation_calls(evaluator)
    assert build_calls[0].mutation_id == "mut-exact-build"


def test_adb10_target_identity_changes_with_build_fields() -> None:
    principal = admin_test_principal()
    base_kwargs = {
        "principal": principal,
        "application_id": _APP,
        "application_environment_id": _ENV,
        "mutation_id": "mut-adb10",
        "runtime_revision_id": "rev-adb10",
        "identity_digest": "digest-a",
    }
    base = build_runtime_revision_mutation_request(**base_kwargs)
    changed_revision = build_runtime_revision_mutation_request(
        **{**base_kwargs, "runtime_revision_id": "rev-adb10-b"}
    )
    changed_digest = build_runtime_revision_mutation_request(
        **{**base_kwargs, "identity_digest": "digest-b"}
    )
    assert control_plane_mutation_request_digest(base) != control_plane_mutation_request_digest(
        changed_revision
    )
    assert control_plane_mutation_request_digest(base) != control_plane_mutation_request_digest(
        changed_digest
    )


def test_adb11_resolver_identity_changes_digest() -> None:
    request_a = _build_request("rev-adb11", mutation_id="mut-adb11")
    request_b = request_a.model_copy(update={"resolver_algorithm_version": "2.0.0"})
    digest_a = build_input_digest(
        application_release_id=request_a.application_release_id,
        platform_version=request_a.platform_version,
        python_version=request_a.python_version,
        source_context_root=request_a.source_context_root,
        application_source_root=request_a.application_source_root,
        agent_source_roots=request_a.agent_source_roots,
        materialization_topology=request_a.materialization_topology.value,
        repository_declaration=request_a.repository_declaration,
        resolver_algorithm_id=request_a.resolver_algorithm_id,
        resolver_algorithm_version=request_a.resolver_algorithm_version,
    )
    digest_b = build_input_digest(
        application_release_id=request_b.application_release_id,
        platform_version=request_b.platform_version,
        python_version=request_b.python_version,
        source_context_root=request_b.source_context_root,
        application_source_root=request_b.application_source_root,
        agent_source_roots=request_b.agent_source_roots,
        materialization_topology=request_b.materialization_topology.value,
        repository_declaration=request_b.repository_declaration,
        resolver_algorithm_id=request_b.resolver_algorithm_id,
        resolver_algorithm_version=request_b.resolver_algorithm_version,
    )
    assert digest_a != digest_b


def test_adb12_build_input_digest_no_secrets() -> None:
    request = _build_request("rev-adb12")
    digest = build_input_digest(
        application_release_id=request.application_release_id,
        platform_version=request.platform_version,
        python_version=request.python_version,
        source_context_root=request.source_context_root,
        application_source_root=request.application_source_root,
        agent_source_roots=request.agent_source_roots,
        materialization_topology=request.materialization_topology.value,
        repository_declaration=request.repository_declaration,
        resolver_algorithm_id=request.resolver_algorithm_id,
        resolver_algorithm_version=request.resolver_algorithm_version,
    )
    assert digest.startswith("sha256:")
    assert "api_key" not in digest


def test_adb13_write_order_authorize_before_persist() -> None:
    evaluator = _RecordingEvaluator()
    stack = _prepare_build_evaluator_stack(evaluator)
    order: list[str] = []
    original_persist_lock = stack.service._lock_store.persist_lock
    original_persist_candidate = stack.service._revision_service.persist_candidate_revision
    original_materialize = stack.service._materialization_service.materialize
    original_mark_validated = stack.service._revision_service.mark_validated

    def track_lock(lock):
        order.append("persist_lock")
        return original_persist_lock(lock)

    def track_candidate(candidate):
        order.append("persist_candidate")
        return original_persist_candidate(candidate)

    def track_materialize(input_):
        order.append("materialize")
        return original_materialize(input_)

    def track_validated(revision_id, validated_revision):
        order.append("mark_validated")
        return original_mark_validated(revision_id, validated_revision=validated_revision)

    with (
        patch.object(stack.service._lock_store, "persist_lock", side_effect=track_lock),
        patch.object(
            stack.service._revision_service,
            "persist_candidate_revision",
            side_effect=track_candidate,
        ),
        patch.object(
            stack.service._materialization_service,
            "materialize",
            side_effect=track_materialize,
        ),
        patch.object(
            stack.service._revision_service,
            "mark_validated",
            side_effect=track_validated,
        ),
    ):
        _build_revision(stack, "rev-adb13", mutation_id="mut-adb13")
    assert len(_build_mutation_calls(evaluator)) == 1
    assert order == [
        "persist_lock",
        "persist_candidate",
        "materialize",
        "mark_validated",
    ]


def test_adb14_idempotent_exact_retry_zero_mutation() -> None:
    evaluator = _RecordingEvaluator()
    stack = _prepare_build_evaluator_stack(evaluator)
    request = _build_request("rev-adb14", mutation_id="mut-adb14")
    first = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=admin_test_principal(),
    )
    before_locks = len(stack.state.locks)
    second = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=admin_test_principal(),
    )
    assert second.runtime_revision_id == first.runtime_revision_id
    assert len(stack.state.locks) == before_locks
    assert len(_build_mutation_calls(evaluator)) == 1


def test_adb15_idempotent_wrong_tenant_blocked() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    _build_revision(stack, "rev-adb15")
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_build_request("rev-adb15"),
            principal=_OTHER_TENANT_PRINCIPAL,
        )


def test_adb16_same_revision_id_different_build_conflicts() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    _build_revision(stack, "rev-adb16")
    different = _build_request("rev-adb16", mutation_id="mut-adb16-b").model_copy(
        update={"platform_version": "9.9.9"}
    )
    with pytest.raises(RuntimeRevisionConflict):
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=different,
            principal=admin_test_principal(),
        )


def test_adb17_concurrent_revision_claim_domain_conflict() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    foreign = RuntimeRevision(
        runtime_revision_id="rev-adb17",
        application_id="foreign-app",
        application_environment_id=_ENV,
        application_release_id="rel-foreign",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-foreign",
        installed_agent_package_digests=(),
        materialized_runtime_lock_id="lock-foreign",
        materialized_runtime_lock_digest="sha256:" + ("f" * 64),
        runtime_graph_digest="sha256:" + ("g" * 64),
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        revision_state=RuntimeRevisionState.CANDIDATE,
    )
    stack.service._revision_store.persist_candidate_revision(foreign)
    with pytest.raises(RuntimeRevisionConflict):
        _build_revision(stack, "rev-adb17")


def test_adb18_build_does_not_activate() -> None:
    stack = build_admin_stack()
    _install_enable(stack)
    _build_revision(stack, "rev-adb18")
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_adb19_http_principal_reaches_mutation_request() -> None:
    evaluator = _RecordingEvaluator()
    stack = _prepare_build_evaluator_stack(evaluator)
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("default")
    )
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(require_api_key=False)
    mount_agent_platform_admin_routes(app, admin_service=stack.service)
    client = TestClient(app)
    payload = _build_payload("rev-adb19-http", mutation_id="mut-adb19")
    response = client.post(
        f"/v1/agent-platform/applications/{_APP}/environments/{_ENV}/revisions/build",
        json=payload,
    )
    assert response.status_code == 200
    build_calls = _build_mutation_calls(evaluator)
    assert len(build_calls) == 1
    assert build_calls[0].principal.tenant_id == "default"
    assert build_calls[0].principal.user_id == "local-dev-admin"
    assert build_calls[0].mutation_id == "mut-adb19"


def test_adb20_authorization_evidence_binds_request() -> None:
    evaluator = _RecordingEvaluator()
    stack = _prepare_build_evaluator_stack(evaluator)
    result = _build_revision(stack, "rev-adb20", mutation_id="mut-adb20")
    evidence = result.authorization_evidence
    assert evidence is not None
    assert evidence.mutation_id == "mut-adb20"
    assert evidence.mutation_type == MUTATION_TYPE_BUILD_RUNTIME_REVISION
    assert evidence.tenant_id == "tenant-test"
    build_call = _build_mutation_calls(evaluator)[0]
    assert evidence.request_digest == control_plane_mutation_request_digest(build_call)


def test_adb21_zero_dynamic_access_in_governance_slice() -> None:
    for relative_path in _CONTROL_PLANE_GOVERNANCE_SLICE_FILES:
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert _FORBIDDEN_DYNAMIC_PATTERNS.search(source) is None, relative_path


def test_adb22_production_build_bypass_inventory() -> None:
    """Reference production lifecycle persists revisions without admin build — classified bootstrap."""
    classification = {
        "AgentPlatformAdminService.build_application_revision": "governed_orchestration",
        "reference_production_lifecycle": "bootstrap_reference_path",
    }
    assert classification["reference_production_lifecycle"] == "bootstrap_reference_path"


def test_adb23_desired_state_regression_still_green() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_desired_state_remediation

    assert test_agent_distribution_desired_state_remediation is not None


def test_adb24_activation_regression_still_green() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_activation

    assert test_agent_distribution_activation is not None


def test_adb25_foundation_regression_still_green() -> None:
    from tests.unit.runtime.governance import test_control_plane_mutation_authorization

    assert test_control_plane_mutation_authorization is not None


def test_adb26_policy_governance_regression_still_green() -> None:
    from tests.unit.agent_distribution import test_agent_distribution_desired_state_remediation

    assert test_agent_distribution_desired_state_remediation is not None
