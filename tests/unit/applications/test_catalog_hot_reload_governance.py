# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R1 catalog hot reload CLA-04 proofs (CHR-1–CHR-12)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.applications._shared.catalog_hot_reload_governance import (
    MUTATION_TYPE_INTEGRATION_CATALOG_HOT_RELOAD,
    build_integration_catalog_hot_reload_mutation_request,
)
from intergrax.applications._shared.catalog_hot_reload_service import (
    BLOCKER_MISSING_BOUNDARY,
    BLOCKER_MISSING_PRINCIPAL,
    BLOCKER_POLICY,
    BLOCKER_POST_AUTH_STALE,
    BLOCKER_PRECONDITION_REVISION,
    CatalogHotReloadService,
)
from intergrax.applications._shared.catalog_hot_reload_wiring import (
    resolve_catalog_hot_reload_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    control_plane_mutation_request_digest,
)
from intergrax.contracts.integration_catalog_hot_reload import CatalogHotReloadOperatorRequest
from intergrax.contracts.integration_catalog_revision import (
    CANONICAL_INTEGRATION_CATALOG_ID,
    CatalogRevision,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog, register_integration, unregister_integration
from intergrax.integrations.registry.catalog_mutation import (
    CatalogReplaceOutcome,
    build_catalog_entries_for_preset,
    current_catalog_revision,
    replace_catalog_if_revision,
)
from intergrax.integrations.registry.catalog_revision import project_target_revision
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationEntry,
    IntegrationStatus,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "product-profile"
_PRINCIPAL = RequestIdentity(
    tenant_id=_TENANT,
    user_id="operator-1",
    principal_type=PrincipalType.USER,
    auth_subject="operator-1",
)


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="catalog.hot_reload.allow",
            decision_id="dec-allow",
        )
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)
    on_evaluate: object | None = None

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        if self.on_evaluate is not None:
            self.on_evaluate()
        return self.decision


def _entry(slug: str) -> IntegrationEntry:
    return IntegrationEntry(
        slug=slug,
        categories=(IntegrationCategory.KEY_VALUE_CACHE,),
        factory=lambda: None,
        status=IntegrationStatus.STABLE,
    )


def _service(evaluator: _RecordingEvaluator) -> CatalogHotReloadService:
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    return CatalogHotReloadService(mutation_authorization_boundary=boundary)


def _request(
    *,
    mutation_id: str = "mut-chr-1",
    preset: str = "core",
) -> CatalogHotReloadOperatorRequest:
    return CatalogHotReloadOperatorRequest(
        mutation_id=mutation_id,
        preset=preset,
        expected_revision=current_catalog_revision(),
    )


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()


def test_chr1_allow_commits_once() -> None:
    evaluator = _RecordingEvaluator()
    service = _service(evaluator)
    before = current_catalog_revision()
    result = service.reload(_request(), principal=_PRINCIPAL)
    assert result.changed is True
    assert result.after_revision.generation == before.generation + 1
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].mutation_type == MUTATION_TYPE_INTEGRATION_CATALOG_HOT_RELOAD


def test_chr2_deny_zero_mutation() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="deny",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="catalog.deny",
            decision_id="dec-deny",
        )
    )
    service = _service(evaluator)
    before = current_catalog_revision()
    result = service.reload(_request(mutation_id="mut-deny"), principal=_PRINCIPAL)
    assert result.changed is False
    assert result.before_revision == before
    assert result.after_revision == before
    assert result.blocker_code == BLOCKER_POLICY


def test_chr3_require_human_zero_mutation_with_scope() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="hitl",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="catalog.hitl",
            decision_id="dec-hitl",
        )
    )
    service = _service(evaluator)
    before = current_catalog_revision()
    result = service.reload(_request(mutation_id="mut-hitl"), principal=_PRINCIPAL)
    assert result.changed is False
    assert current_catalog_revision() == before
    assert result.authorization_evidence is not None
    assert result.authorization_evidence.policy_action is PolicyAction.REQUIRE_HUMAN


def test_chr4_escalate_zero_mutation() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.ESCALATE,
            reason="escalate",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="catalog.escalate",
            decision_id="dec-esc",
        )
    )
    service = _service(evaluator)
    before = current_catalog_revision()
    result = service.reload(_request(mutation_id="mut-esc"), principal=_PRINCIPAL)
    assert result.changed is False
    assert current_catalog_revision() == before


def test_chr5_evaluator_failure_zero_mutation() -> None:
    class _FailingEvaluator:
        def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
            raise RuntimeError("evaluator boom")

    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_FailingEvaluator())
    service = CatalogHotReloadService(mutation_authorization_boundary=boundary)
    before = current_catalog_revision()
    result = service.reload(_request(mutation_id="mut-fail"), principal=_PRINCIPAL)
    assert result.changed is False
    assert current_catalog_revision() == before
    assert result.policy_action == str(PolicyAction.DENY)


def test_chr6_stale_revision_after_authorization_blocks_commit() -> None:
    def _mutate_during_auth() -> None:
        register_integration(_entry("stale-interrupt"), override=True)

    evaluator = _RecordingEvaluator(on_evaluate=_mutate_during_auth)
    service = _service(evaluator)
    result = service.reload(_request(mutation_id="mut-stale"), principal=_PRINCIPAL)
    assert result.changed is False
    assert result.blocker_code == BLOCKER_POST_AUTH_STALE


def test_chr7_request_digest_changes_with_target() -> None:
    before = current_catalog_revision()
    candidate = build_catalog_entries_for_preset("core")
    target = project_target_revision(before, candidate)
    base = build_integration_catalog_hot_reload_mutation_request(
        mutation_id="mut-d1",
        principal=_PRINCIPAL,
        preset="core",
        current_revision=before,
        target_revision=target,
    )
    other_target = project_target_revision(
        before,
        {**candidate, "extra": _entry("extra-slug")},
    )
    other = build_integration_catalog_hot_reload_mutation_request(
        mutation_id="mut-d1",
        principal=_PRINCIPAL,
        preset="core",
        current_revision=before,
        target_revision=other_target,
    )
    assert control_plane_mutation_request_digest(base) != control_plane_mutation_request_digest(
        other
    )


def test_chr8_same_material_mutation_stable_digest() -> None:
    before = current_catalog_revision()
    candidate = build_catalog_entries_for_preset("core")
    target = project_target_revision(before, candidate)
    first = build_integration_catalog_hot_reload_mutation_request(
        mutation_id="mut-s1",
        principal=_PRINCIPAL,
        preset="core",
        current_revision=before,
        target_revision=target,
    )
    second = build_integration_catalog_hot_reload_mutation_request(
        mutation_id="mut-s1",
        principal=_PRINCIPAL,
        preset="core",
        current_revision=before,
        target_revision=target,
    )
    assert control_plane_mutation_request_digest(first) == control_plane_mutation_request_digest(
        second
    )


def test_chr9_same_state_reload_no_op_without_generation_bump() -> None:
    candidate = build_catalog_entries_for_preset("core")
    committed = replace_catalog_if_revision(
        expected_revision=current_catalog_revision(),
        candidate_entries=candidate,
    )
    after_first = committed.after_revision
    evaluator = _RecordingEvaluator()
    service = _service(evaluator)
    result = service.reload(
        CatalogHotReloadOperatorRequest(
            mutation_id="mut-noop",
            preset="core",
            expected_revision=after_first,
        ),
        principal=_PRINCIPAL,
    )
    assert result.changed is False
    assert result.after_revision == after_first


def test_chr10_external_evaluator_receives_exact_request() -> None:
    evaluator = _RecordingEvaluator()
    service = _service(evaluator)
    service.reload(_request(mutation_id="mut-plugin"), principal=_PRINCIPAL)
    req = evaluator.calls[0]
    assert req.resource_id == CANONICAL_INTEGRATION_CATALOG_ID
    assert req.resource_type == "integration_catalog"


def test_chr11_product_enabled_missing_boundary_fail_closed() -> None:
    wiring = resolve_catalog_hot_reload_wiring(ApplicationEnvironmentProfile.product_defaults())
    assert wiring.enabled is True
    assert wiring.service is not None
    result = wiring.service.reload(
        _request(mutation_id="mut-noboundary"),
        principal=_PRINCIPAL,
    )
    assert result.blocker_code == BLOCKER_MISSING_BOUNDARY
    assert result.changed is False


def test_chr12_composition_does_not_execute_reload_automatically() -> None:
    before = current_catalog_revision()
    resolve_catalog_hot_reload_wiring(ApplicationEnvironmentProfile.product_defaults())
    assert current_catalog_revision() == before


def test_chr_r1r1_1_direct_register_invalidates_stale_hot_reload_cas() -> None:
    at_authorization = current_catalog_revision()
    register_integration(_entry("direct-mutation"), override=True)
    result = replace_catalog_if_revision(
        expected_revision=at_authorization,
        candidate_entries=build_catalog_entries_for_preset("core"),
    )
    assert result.outcome is CatalogReplaceOutcome.REVISION_CONFLICT


def test_chr_r1r1_2_unregister_invalidates_stale_hot_reload_cas() -> None:
    register_integration(_entry("ephemeral"), override=True)
    at_authorization = current_catalog_revision()
    unregister_integration("ephemeral")
    result = replace_catalog_if_revision(
        expected_revision=at_authorization,
        candidate_entries={"ephemeral": _entry("ephemeral")},
    )
    assert result.outcome is CatalogReplaceOutcome.REVISION_CONFLICT


def test_chr_r1r1_3_aba_old_authorization_rejected_on_hot_reload() -> None:
    register_integration(_entry("alpha"), override=True)
    authorized = current_catalog_revision()
    register_integration(_entry("beta"), override=True)
    unregister_integration("beta")
    evaluator = _RecordingEvaluator()
    service = _service(evaluator)
    result = service.reload(
        CatalogHotReloadOperatorRequest(
            mutation_id="mut-aba",
            preset="core",
            expected_revision=authorized,
        ),
        principal=_PRINCIPAL,
    )
    assert result.changed is False
    assert result.blocker_code == BLOCKER_PRECONDITION_REVISION


def test_chr_r1r1_4_explicit_principal_reaches_cla04_request() -> None:
    evaluator = _RecordingEvaluator()
    service = _service(evaluator)
    service.reload(_request(mutation_id="mut-principal"), principal=_PRINCIPAL)
    req = evaluator.calls[0]
    assert req.principal.tenant_id == _PRINCIPAL.tenant_id
    assert req.principal.user_id == _PRINCIPAL.user_id
    assert req.principal.auth_subject == _PRINCIPAL.auth_subject
    assert req.principal.principal_type == _PRINCIPAL.principal_type


def test_chr_r1r1_5_missing_principal_fails_closed() -> None:
    service = _service(_RecordingEvaluator())
    result = service.reload(_request(mutation_id="mut-noprincipal"))
    assert result.changed is False
    assert result.blocker_code == BLOCKER_MISSING_PRINCIPAL


def test_chr_r1r1_6_composition_does_not_synthesize_request_identity() -> None:
    from pathlib import Path

    wiring_path = (
        Path(__file__).resolve().parents[3]
        / "intergrax/applications/_shared/catalog_hot_reload_wiring.py"
    )
    source = wiring_path.read_text(encoding="utf-8")
    assert "RequestIdentity(" not in source


def test_chr_r1r1_7_external_evaluator_still_receives_exact_request() -> None:
    test_chr10_external_evaluator_receives_exact_request()


def test_chr_r1r1_8_composition_does_not_mutate_catalog() -> None:
    test_chr12_composition_does_not_execute_reload_automatically()
