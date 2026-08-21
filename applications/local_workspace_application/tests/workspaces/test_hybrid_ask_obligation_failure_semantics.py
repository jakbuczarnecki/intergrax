# © Artur Czarnecki. All rights reserved.

"""COMM-5 F3-E obligation-level failure semantics integration tests."""

from __future__ import annotations

pytest_plugins = [
    "local_workspace_application.tests.workspaces.test_hybrid_ask_multi_provider",
]

from datetime import timedelta

import httpx
import pytest

from local_workspace_application.tests.workspaces.test_hybrid_ask_multi_provider import (
    _BINDING_GOVERNANCE,
    _FixedPolicyRulesPort,
    _MutableConfigurationService,
    _NOW,
    _RevokingOrchestrator,
    _TemporalPolicyRulesPort,
    _TENANT,
    _WORKSPACE,
    _all_success_bindings,
    _build_service,
    _configuration,
    _register_connections,
    _seed_fresh_provider_snapshots,
)
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityStatusV1,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    KnowledgeQueryAudienceV1,
)
from local_workspace_application.workspaces.hybrid_ask_service import WorkspaceAskCommandV2
from local_workspace_application.workspaces.knowledge_configuration_models import QueryPolicyModeV2
from proof_infrastructure.controlled_change_approval_service.lifecycle import (
    ControlledChangeApprovalServer,
)
from proof_infrastructure.controlled_governance_approval_service.lifecycle import (
    ControlledGovernanceApprovalServer,
)
from proof_infrastructure.controlled_project_status_service.lifecycle import (
    ControlledProjectStatusServer,
)
from proof_infrastructure.controlled_project_status_service.models import ProjectBlockerStatusV1
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
)
from proof_infrastructure.controlled_security_status_service.lifecycle import (
    ControlledSecurityStatusServer,
)
from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorV1,
)

pytestmark = pytest.mark.unit


def _evaluation_for_suffix(run, suffix: str):
    assert run.evidence_admissibility is not None
    return next(
        item
        for item in run.evidence_admissibility.requirement_evaluations
        if item.requirement_id.endswith(f":{suffix}")
    )


def _seed_readiness(project_status_server: ControlledProjectStatusServer) -> None:
    httpx.put(
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


@pytest.mark.asyncio
async def test_authority_unavailable_preserves_other_provider_evidence(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
    from intergrax.runtime.evidence.obligation_derivation import (
        DeterministicEvidenceObligationDerivation,
    )
    from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
    from intergrax.runtime.vendor_knowledge.live.bootstrap import (
        build_vendor_knowledge_live_registration_registry,
    )
    from intergrax.runtime.vendor_knowledge.provider_composition import (
        build_default_vendor_knowledge_connection_factory_registry,
    )
    from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
        DocumentStoreTenantConnectionRepository,
    )
    from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
        TenantConnectionRehydrator,
    )
    from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
        RepositoryTenantConnectionPort,
    )
    from local_workspace_application.tests.workspaces.test_hybrid_ask_multi_provider import (
        _Catalog,
        _EmptyIndexedRetriever,
        _RecordingLLM,
        _RecordingSecretsStore,
        _Repository,
        _WorkspaceAuthority,
    )
    from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
    from local_workspace_application.workspaces.hybrid_ask_execution import (
        KnowledgeConnectionRegistryIntegrationResolverV1,
        KnowledgeQueryOrchestratorV1,
        LiveCapabilityExecutorV1,
    )
    from local_workspace_application.workspaces.hybrid_ask_service import (
        BindingResourceScopeValidator,
        SafeCapabilityRequestEnvelopeValidator,
        WorkspaceAskServiceV2,
    )
    from local_workspace_application.workspaces.knowledge_live_access_service import (
        WorkspaceLiveAccessRuntimeAuthority,
    )

    _seed_readiness(project_status_server)
    bindings = _all_success_bindings()
    configuration_service = _MutableConfigurationService(_configuration(bindings=bindings))
    document_store = InMemoryDocumentStore()
    connection_repository = DocumentStoreTenantConnectionRepository(document_store)
    connections = _register_connections(
        connection_repository,
        project_status_url=project_status_server.base_url,
        security_status_url=security_status_server.base_url,
        change_approval_url=change_approval_server.base_url,
        governance_approval_url=governance_approval_server.base_url,
    )
    connection_registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=_RecordingSecretsStore(),
        integration_factory=build_default_vendor_knowledge_connection_factory_registry(),
        connection_registry=connection_registry,
    )
    for connection in connections:
        rehydrator.rehydrate_connection(
            tenant_id=_TENANT,
            connection_ref=connection.connection_ref,
        )
    published = build_vendor_knowledge_live_registration_registry().publish()
    authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=configuration_service,  # type: ignore[arg-type]
        tenant_connection_port=RepositoryTenantConnectionPort(connection_repository),
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
    )
    executor = LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=KnowledgeConnectionRegistryIntegrationResolverV1(
            connection_registry
        ),
        runtime_authority=authority,
        clock=lambda: _NOW,
    )
    inner_orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=_EmptyIndexedRetriever(),  # type: ignore[arg-type]
        live_executor=executor,
        clock=lambda: _NOW,
    )
    orchestrator = _RevokingOrchestrator(
        inner=inner_orchestrator,
        configuration_service=configuration_service,
        binding_id=_BINDING_GOVERNANCE,
    )
    governance_approval_server.store.reset_read_request_count()
    llm = _RecordingLLM()
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=_Repository(configuration_service._configuration),  # type: ignore[arg-type]
        ask_repository=WorkspaceAskRepository(document_store),
        configuration_service=configuration_service,  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
        request_envelope_validator=SafeCapabilityRequestEnvelopeValidator(
            schema_registry=published.schemas,
        ),
        resource_scope_validator=BindingResourceScopeValidator(),
        orchestrator=orchestrator,  # type: ignore[arg-type]
        llm_adapter=llm,
        clock=lambda: _NOW,
        run_id_factory=lambda: "run-f3e-authority",
        plan_id_factory=lambda: "plan-f3e-authority",
        evidence_obligation_derivation_port=DeterministicEvidenceObligationDerivation(),
        resolved_policy_rules_port=_FixedPolicyRulesPort(),
        schema_registry=published.schemas,
    )

    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-f3e-authority",
            request_id="request-f3e-authority",
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert governance_approval_server.store.read_request_count() == 0
    governance_eval = _evaluation_for_suffix(run, "architecture")
    assert governance_eval.reason_code is RequirementAdmissibilityReasonCodeV1.AUTHORITY_UNAVAILABLE
    readiness_eval = _evaluation_for_suffix(run, "readiness")
    assert readiness_eval.status is RequirementEvaluationStatusV1.SATISFIED


@pytest.mark.asyncio
async def test_provider_failed_isolated_to_security_obligation(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    _seed_readiness(project_status_server)
    httpx.put(
        f"{security_status_server.base_url}/control/read-behavior",
        json={"behavior": SecurityStatusReadBehaviorV1.HTTP_503.value},
        timeout=2.0,
    )
    security_status_server.store.reset_read_request_count()

    service, llm, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
    )
    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-f3e-provider-failed",
            request_id="request-f3e-provider-failed",
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert security_status_server.store.read_request_count() == 1
    security_eval = _evaluation_for_suffix(run, "security")
    assert security_eval.reason_code is RequirementAdmissibilityReasonCodeV1.PROVIDER_FAILED
    readiness_eval = _evaluation_for_suffix(run, "readiness")
    assert readiness_eval.status is RequirementEvaluationStatusV1.SATISFIED
    assert len(run.live_call_failures) == 1
    assert run.live_call_failures[0].call_id.endswith(":security-read")


@pytest.mark.asyncio
async def test_provider_response_invalid_for_malformed_security_payload(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    _seed_readiness(project_status_server)
    httpx.put(
        f"{security_status_server.base_url}/control/read-behavior",
        json={"behavior": SecurityStatusReadBehaviorV1.MALFORMED_JSON.value},
        timeout=2.0,
    )
    security_status_server.store.reset_read_request_count()

    service, llm, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
    )
    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-f3e-invalid-response",
            request_id="request-f3e-invalid-response",
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert security_status_server.store.read_request_count() == 1
    security_eval = _evaluation_for_suffix(run, "security")
    assert (
        security_eval.reason_code
        is RequirementAdmissibilityReasonCodeV1.PROVIDER_RESPONSE_INVALID
    )


@pytest.mark.asyncio
async def test_temporal_invalid_regression_for_security_obligation(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    await _seed_fresh_provider_snapshots(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        security_updated_at=_NOW - timedelta(hours=6),
    )
    service, llm, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
        resolved_policy_rules_port=_TemporalPolicyRulesPort(
            security_max_age_seconds=3_600,
        ),
    )
    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-f3e-temporal-invalid",
            request_id="request-f3e-temporal-invalid",
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    security_eval = _evaluation_for_suffix(run, "security")
    assert (
        security_eval.reason_code
        is RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID
    )
    assert run.live_call_failures == ()


@pytest.mark.asyncio
async def test_all_success_has_no_failure_reasons_and_llm(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    await _seed_fresh_provider_snapshots(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        security_updated_at=_NOW - timedelta(minutes=30),
    )
    service, llm, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
        resolved_policy_rules_port=_TemporalPolicyRulesPort(
            security_max_age_seconds=3_600,
        ),
    )
    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-f3e-all-success",
            request_id="request-f3e-all-success",
        )
    )

    assert run.status is AskRunStatus.COMPLETED
    assert llm.calls == 1
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )
    assert all(
        item.reason_code is None
        for item in run.evidence_admissibility.requirement_evaluations
    )
    assert run.live_call_failures == ()


@pytest.mark.asyncio
async def test_run_reload_preserves_failure_semantics(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    _seed_readiness(project_status_server)
    httpx.put(
        f"{security_status_server.base_url}/control/read-behavior",
        json={"behavior": SecurityStatusReadBehaviorV1.HTTP_503.value},
        timeout=2.0,
    )

    service, _, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
    )
    await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-f3e-reload",
            request_id="request-f3e-reload",
        )
    )
    reloaded = service.get_run(tenant_id=_TENANT, run_id="run-f3e-reload")
    security_eval = _evaluation_for_suffix(reloaded, "security")
    assert security_eval.reason_code is RequirementAdmissibilityReasonCodeV1.PROVIDER_FAILED
    assert len(reloaded.live_call_failures) == 1
