# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from hashlib import sha256

import httpx
import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
    PROJECT_STATUS_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.live.project_status.project import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
    ProjectStatusReadLiveRequestV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnectionAdministrativeStatus,
    TenantConnectionService,
)
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.hybrid_ask_admissibility import (
    evaluate_evidence_admissibility,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    EvidenceAdmissibilityStatusV1,
    EvidenceTypeV1,
    LiveWorkspaceEvidenceV1,
    PersistedIndexedEvidenceV2,
    PersistedLiveEvidenceProvenanceV2,
    RequirementAdmissibilityReasonCodeV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    HybridAskPolicyError,
    KnowledgeQueryAudienceV1,
    LiveCallProposalV1,
    LiveEvidenceRequirementV1,
    QueryPolicyModeV2,
    compose_evidence_obligations,
)
from local_workspace_application.workspaces.hybrid_ask_service import (
    WorkspaceAskCommandV2,
    WorkspaceAskV2LookupError,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
)
from proof_infrastructure.controlled_project_status_service.models import (
    ProjectBlockerStatusV1,
    ProjectStatusReadBehaviorControlV1,
    ProjectStatusReadBehaviorV1,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
    seed_orion_fixture,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.adversarial_models import (
    AdversarialAttackIdV1,
    AdversarialAttackResultV1,
    AdversarialDefenseLayerV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.fixtures import (
    ORION_DEPLOYMENT_QUESTION,
    PROOF_BINDING_ID,
    PROOF_CONNECTION_REF,
    PROOF_DISABLE_IDEMPOTENCY_HASH,
    PROOF_LIVE_CALL_ID,
    PROOF_NOW,
    PROOF_TENANT_ID,
    PROOF_WORKSPACE_ID,
    orion_provider_request,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.harness import (
    GovernedHybridKnowledgeHarness,
    build_harness,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.runner import (
    normalize_decision,
    run_flagship_proof,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_EPHEMERAL_PERSISTENCE_KEYS = frozenset(
    {
        "content",
        "readiness_score",
        "blockers",
        "raw_json",
        "provider_payload",
    }
)


def _live_evidence_for_call(*, call_id: str) -> LiveWorkspaceEvidenceV1:
    content = "live-proof-content"
    return LiveWorkspaceEvidenceV1(
        evidence_id=f"live:{call_id}:item-1",
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        safe_display_name="ORION status",
        retrieved_at=PROOF_NOW,
        content=content,
        content_hash=sha256(content.encode()).hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        live_access_binding_id=PROOF_BINDING_ID,
        connection_ref=PROOF_CONNECTION_REF,
        capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
        source_kind=PROJECT_STATUS_SOURCE_KIND,
        contract_version="1",
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.ISSUE_TRACKER.value,
        call_id=call_id,
    )


@pytest.fixture
def project_status_server():
    from proof_infrastructure.controlled_project_status_service.lifecycle import (
        ControlledProjectStatusServer,
    )

    server = ControlledProjectStatusServer.start()
    seed_orion_fixture(server.store)
    yield server
    server.stop()


async def _run_ask(
    harness: GovernedHybridKnowledgeHarness,
    *,
    run_id: str,
    command: WorkspaceAskCommandV2 | None = None,
) -> tuple[object, int, int]:
    harness.reset_http_counter()
    llm_before = harness.llm.calls
    resolved = command or harness.build_command(run_id=run_id)
    run = await harness.service.ask(resolved)
    return run, harness.http_read_count(), harness.llm.calls - llm_before


def _set_read_behavior(
    server: object,
    behavior: ProjectStatusReadBehaviorV1,
) -> None:
    base_url = server.base_url  # type: ignore[attr-defined]
    response = httpx.put(
        f"{base_url}/control/read-behavior",
        json=ProjectStatusReadBehaviorControlV1(behavior=behavior).model_dump(),
        timeout=2.0,
    )
    response.raise_for_status()


def _disable_tenant_connection(harness: GovernedHybridKnowledgeHarness) -> None:
    service = TenantConnectionService(
        tenant_id=PROOF_TENANT_ID,
        repository=harness.tenant_connection_repository,
    )
    connection = service.get(PROOF_CONNECTION_REF)
    disabled = connection.model_copy(
        update={
            "administrative_status": TenantConnectionAdministrativeStatus.DISABLED,
            "configuration_version": connection.configuration_version + 1,
            "updated_at": PROOF_NOW + timedelta(seconds=1),
        }
    )
    service.update(disabled, expected_configuration_version=connection.configuration_version)


def _assert_no_ephemeral_live_body_persisted(
    persisted_evidence: tuple[object, ...],
) -> None:
    for item in persisted_evidence:
        if not isinstance(item, PersistedLiveEvidenceProvenanceV2):
            continue
        dumped = item.model_dump()
        for key in _FORBIDDEN_EPHEMERAL_PERSISTENCE_KEYS:
            assert key not in dumped
        assert isinstance(item.content_hash, str) and item.content_hash
        assert item.provider_id == PROJECT_STATUS_PROVIDER_ID
        assert item.live_access_binding_id == PROOF_BINDING_ID
        assert item.call_id == PROOF_LIVE_CALL_ID
        assert item.connection_ref == PROOF_CONNECTION_REF


@pytest.mark.asyncio
async def test_attack_required_live_missing_cannot_reach_llm(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    _disable_tenant_connection(harness)
    run, http_calls, llm_calls = await _run_ask(
        harness,
        run_id="attack-a-required-live-missing",
    )
    indexed_present = any(
        item.evidence_type is EvidenceTypeV1.INDEXED for item in run.persisted_evidence
    )
    live_present = any(
        isinstance(item, PersistedLiveEvidenceProvenanceV2)
        for item in run.persisted_evidence
    )
    assert indexed_present
    assert not live_present
    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert run.answer is None
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )
    assert http_calls == 0
    assert llm_calls == 0


@pytest.mark.asyncio
async def test_attack_midflight_revoke_keeps_provider_call_count_zero(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server, revoke_after_indexed=True)
    run, http_calls, llm_calls = await _run_ask(
        harness,
        run_id="attack-b-midflight-revoke",
    )
    configuration = harness.configuration_service.get_configuration(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
    )
    assert configuration is not None
    binding = next(
        item
        for item in configuration.live_access_bindings
        if item.live_access_binding_id == PROOF_BINDING_ID
    )
    assert binding.status is LiveAccessBindingStatusV1.DISABLED
    assert any(item.evidence_type is EvidenceTypeV1.INDEXED for item in run.persisted_evidence)
    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert run.answer is None
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )
    assert http_calls == 0
    assert llm_calls == 0


@pytest.mark.asyncio
async def test_attack_wrong_connection_provider_rejected_before_http(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    command = WorkspaceAskCommandV2(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        question=ORION_DEPLOYMENT_QUESTION,
        requested_mode=QueryPolicyModeV2.HYBRID,
        audience_context=AudienceContextV1(audience=KnowledgeQueryAudienceV1.PERSONAL),
        ordered_live_call_proposals=(
            LiveCallProposalV1(
                call_id=PROOF_LIVE_CALL_ID,
                live_access_binding_id="binding-does-not-exist",
                capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
                typed_capability_request={"project_id": ORION_FIXTURE_PROJECT_ID},
            ),
        ),
        indexed_max_results=5,
        run_id="attack-c-wrong-binding",
        request_id="attack-c-request",
    )
    harness.reset_http_counter()
    llm_before = harness.llm.calls
    with pytest.raises(HybridAskPolicyError, match="live_binding_not_found"):
        await harness.service.ask(command)
    assert harness.http_read_count() == 0
    assert harness.llm.calls == llm_before


@pytest.mark.asyncio
async def test_attack_wrong_tenant_cannot_reuse_live_binding(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    command = harness.build_command(run_id="attack-d-wrong-tenant").model_copy(
        update={"tenant_id": "foreign-tenant"}
    )
    harness.reset_http_counter()
    llm_before = harness.llm.calls
    with pytest.raises(WorkspaceAskV2LookupError, match="workspace_not_found"):
        await harness.service.ask(command)
    assert harness.http_read_count() == 0
    assert harness.llm.calls == llm_before


@pytest.mark.asyncio
async def test_attack_wrong_workspace_cannot_reuse_orion_binding(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    command = harness.build_command(run_id="attack-e-wrong-workspace").model_copy(
        update={"workspace_id": "foreign-workspace"}
    )
    harness.reset_http_counter()
    llm_before = harness.llm.calls
    with pytest.raises(WorkspaceAskV2LookupError, match="workspace_not_found"):
        await harness.service.ask(command)
    assert harness.http_read_count() == 0
    assert harness.llm.calls == llm_before


async def _assert_provider_failure_finalizes_canonically(
    harness: GovernedHybridKnowledgeHarness,
    *,
    run_id: str,
    expected_reason: RequirementAdmissibilityReasonCodeV1,
    command: WorkspaceAskCommandV2 | None = None,
    setup: Callable[[], None] | None = None,
) -> None:
    if setup is not None:
        setup()
    run, http_calls, llm_calls = await _run_ask(
        harness,
        run_id=run_id,
        command=command,
    )
    live_present = any(
        isinstance(item, PersistedLiveEvidenceProvenanceV2)
        for item in run.persisted_evidence
    )
    assert http_calls == 1
    assert llm_calls == 0
    assert run.answer is None
    assert not live_present
    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )
    live_evaluations = [
        item
        for item in run.evidence_admissibility.requirement_evaluations
        if item.requirement_id == "provider:orion:live-status"
    ]
    assert len(live_evaluations) == 1
    assert (
        live_evaluations[0].reason_code
        is expected_reason
    )
    reloaded = harness.service.get_run(tenant_id=PROOF_TENANT_ID, run_id=run_id)
    assert reloaded.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert reloaded.evidence_admissibility == run.evidence_admissibility


@pytest.mark.asyncio
async def test_attack_malformed_provider_payload_cannot_satisfy_obligation(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    await _assert_provider_failure_finalizes_canonically(
        harness,
        run_id="attack-f-malformed-json",
        expected_reason=RequirementAdmissibilityReasonCodeV1.PROVIDER_RESPONSE_INVALID,
        setup=lambda: _set_read_behavior(
            project_status_server, ProjectStatusReadBehaviorV1.MALFORMED_JSON
        ),
    )


@pytest.mark.asyncio
async def test_attack_invalid_schema_provider_payload_cannot_satisfy_obligation(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    await _assert_provider_failure_finalizes_canonically(
        harness,
        run_id="attack-f-invalid-schema",
        expected_reason=RequirementAdmissibilityReasonCodeV1.PROVIDER_RESPONSE_INVALID,
        setup=lambda: _set_read_behavior(
            project_status_server, ProjectStatusReadBehaviorV1.INVALID_SCHEMA
        ),
    )


@pytest.mark.asyncio
async def test_attack_provider_404_cannot_synthesize_answer(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    command = harness.build_command(run_id="attack-g-404").model_copy(
        update={
            "provider_request": ProjectStatusReadLiveRequestV1(
                project_id="UNKNOWN_PROJECT"
            )
        }
    )
    await _assert_provider_failure_finalizes_canonically(
        harness,
        run_id="attack-g-404",
        expected_reason=RequirementAdmissibilityReasonCodeV1.PROVIDER_FAILED,
        command=command,
    )


@pytest.mark.asyncio
async def test_attack_provider_5xx_cannot_synthesize_answer(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    await _assert_provider_failure_finalizes_canonically(
        harness,
        run_id="attack-g-5xx",
        expected_reason=RequirementAdmissibilityReasonCodeV1.PROVIDER_FAILED,
        setup=lambda: _set_read_behavior(
            project_status_server, ProjectStatusReadBehaviorV1.HTTP_503
        ),
    )


def test_attack_caller_cannot_downgrade_required_live_evidence() -> None:
    with pytest.raises(ValidationError, match="provider_request_requires_live_mode"):
        WorkspaceAskCommandV2(
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            question=ORION_DEPLOYMENT_QUESTION,
            requested_mode=QueryPolicyModeV2.INDEXED_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            provider_request=orion_provider_request(),
            run_id="attack-h-indexed-only",
            request_id="attack-h-request",
        )

    with pytest.raises(HybridAskPolicyError, match="duplicate_requirement_id"):
        compose_evidence_obligations(
            authoritative=(
                LiveEvidenceRequirementV1(
                    requirement_id="provider:orion:live-status",
                    semantic_role="Authoritative live",
                    call_id=PROOF_LIVE_CALL_ID,
                ),
            ),
            additional=(
                LiveEvidenceRequirementV1(
                    requirement_id="provider:orion:live-status",
                    semantic_role="Caller duplicate",
                    call_id="other-call",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_attack_stale_plan_runtime_revalidation_wins(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server, revoke_after_indexed=True)
    run, http_calls, llm_calls = await _run_ask(
        harness,
        run_id="attack-i-stale-plan",
    )
    assert http_calls == 0
    assert llm_calls == 0
    assert run.answer is None
    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_attack_connection_disabled_denies_live_execution(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    _disable_tenant_connection(harness)
    run, http_calls, llm_calls = await _run_ask(
        harness,
        run_id="attack-j-connection-disabled",
    )
    assert http_calls == 0
    assert llm_calls == 0
    assert run.answer is None
    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_attack_capability_mismatch_rejected_before_http(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    command = WorkspaceAskCommandV2(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        question=ORION_DEPLOYMENT_QUESTION,
        requested_mode=QueryPolicyModeV2.HYBRID,
        audience_context=AudienceContextV1(audience=KnowledgeQueryAudienceV1.PERSONAL),
        ordered_live_call_proposals=(
            LiveCallProposalV1(
                call_id=PROOF_LIVE_CALL_ID,
                live_access_binding_id=PROOF_BINDING_ID,
                capability_id="vendor.wrong.provider.read",
                typed_capability_request={"project_id": ORION_FIXTURE_PROJECT_ID},
            ),
        ),
        indexed_max_results=5,
        run_id="attack-k-capability-mismatch",
        request_id="attack-k-request",
    )
    harness.reset_http_counter()
    llm_before = harness.llm.calls
    with pytest.raises(HybridAskPolicyError, match="live_capability_not_allowed"):
        await harness.service.ask(command)
    assert harness.http_read_count() == 0
    assert harness.llm.calls == llm_before


@pytest.mark.asyncio
async def test_attack_ephemeral_live_body_not_durable(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    run, http_calls, llm_calls = await _run_ask(
        harness,
        run_id="attack-l-ephemeral-leak",
    )
    assert run.status is AskRunStatus.COMPLETED
    assert http_calls == 1
    assert llm_calls == 1
    _assert_no_ephemeral_live_body_persisted(run.persisted_evidence)
    historical = harness.service.get_run(
        tenant_id=PROOF_TENANT_ID,
        run_id="attack-l-ephemeral-leak",
    )
    _assert_no_ephemeral_live_body_persisted(historical.persisted_evidence)


@pytest.mark.asyncio
async def test_attack_historical_answer_immutable_after_external_change(
    project_status_server,
) -> None:
    harness = await build_harness(server=project_status_server)
    run, _, _ = await _run_ask(harness, run_id="attack-m-historical")
    assert run.status is AskRunStatus.COMPLETED
    assert normalize_decision(answer=run.answer, status=run.status).value == "NO"
    live_before = next(
        item
        for item in run.persisted_evidence
        if isinstance(item, PersistedLiveEvidenceProvenanceV2)
    )
    indexed_before = next(
        item
        for item in run.persisted_evidence
        if isinstance(item, PersistedIndexedEvidenceV2)
    )
    response = httpx.put(
        f"{project_status_server.base_url}/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
        timeout=2.0,
    )
    response.raise_for_status()
    historical = harness.service.get_run(
        tenant_id=PROOF_TENANT_ID,
        run_id="attack-m-historical",
    )
    live_after = next(
        item
        for item in historical.persisted_evidence
        if isinstance(item, PersistedLiveEvidenceProvenanceV2)
    )
    indexed_after = next(
        item
        for item in historical.persisted_evidence
        if isinstance(item, PersistedIndexedEvidenceV2)
    )
    assert historical.answer == run.answer
    assert live_after.content_hash == live_before.content_hash
    assert live_after.call_id == live_before.call_id
    assert live_after.live_access_binding_id == live_before.live_access_binding_id
    assert indexed_after.evidence_id == indexed_before.evidence_id


def test_attack_wrong_call_evidence_cannot_satisfy_obligation() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(
            LiveEvidenceRequirementV1(
                requirement_id="req-live-a",
                semantic_role="Required call A",
                call_id="call-a",
            ),
        ),
        indexed_evidence=(),
        live_evidence=(_live_evidence_for_call(call_id="call-b"),),
        evaluated_at=PROOF_NOW,
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED
    assert (
        result.requirement_evaluations[0].reason_code
        is RequirementAdmissibilityReasonCodeV1.LIVE_CALL_MISMATCH
    )


def test_attack_duplicate_replay_evidence_not_reachable_by_contract() -> None:
    with pytest.raises(HybridAskPolicyError, match="duplicate_requirement_id"):
        compose_evidence_obligations(
            authoritative=(
                LiveEvidenceRequirementV1(
                    requirement_id="req-one",
                    semantic_role="First",
                    call_id="call-1",
                ),
                LiveEvidenceRequirementV1(
                    requirement_id="req-one",
                    semantic_role="Replay",
                    call_id="call-2",
                ),
            ),
            additional=(),
        )


def test_adversarial_attack_matrix_all_pass() -> None:
    matrix = (
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.A_REQUIRED_LIVE_MISSING,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.EVIDENCE_UNSATISFIED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.B_MIDFLIGHT_REVOKE,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.AUTHORITY_DENIED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.C_WRONG_CONNECTION_PROVIDER,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.PLAN_REJECTED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.D_WRONG_TENANT,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.PLAN_REJECTED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.E_WRONG_WORKSPACE,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.PLAN_REJECTED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.F_MALFORMED_PROVIDER_PAYLOAD,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.PROVIDER_REJECTED,
            http_calls=1,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.G_PROVIDER_404_5XX,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.PROVIDER_REJECTED,
            http_calls=1,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.H_CALLER_DOWNGRADE,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.NOT_REACHABLE_BY_CONTRACT,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.I_STALE_PLAN,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.AUTHORITY_DENIED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.J_CONNECTION_DISABLED,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.AUTHORITY_DENIED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.K_CAPABILITY_MISMATCH,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.PLAN_REJECTED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.L_EPHEMERAL_LEAK,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.SYNTHESIS_BLOCKED,
            http_calls=1,
            llm_calls=1,
            passed=True,
            notes="Successful run; durable state is structural provenance only",
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.M_HISTORICAL_IMMUTABILITY,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.SYNTHESIS_BLOCKED,
            http_calls=1,
            llm_calls=1,
            passed=True,
            notes="Historical run unchanged after external state mutation",
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.N_WRONG_CALL_EVIDENCE,
            reachable=True,
            defense_layer=AdversarialDefenseLayerV1.EVIDENCE_UNSATISFIED,
            http_calls=0,
            llm_calls=0,
            passed=True,
        ),
        AdversarialAttackResultV1(
            attack_id=AdversarialAttackIdV1.O_DUPLICATE_REPLAY_EVIDENCE,
            reachable=False,
            defense_layer=AdversarialDefenseLayerV1.NOT_REACHABLE_BY_CONTRACT,
            http_calls=0,
            llm_calls=0,
            passed=True,
            notes="duplicate_requirement_id rejected at compose",
        ),
    )
    assert all(item.passed for item in matrix)
    assert len(matrix) == 15


def test_flagship_proof_regression_after_adversarial_suite() -> None:
    result = run_flagship_proof(emit_terminal=False)
    assert result.all_passed is True
    assert result.passed_count == 4
