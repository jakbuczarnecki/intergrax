# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    EvidenceObligationDerivationContextV1,
    PolicyRevisionReferenceV1,
    RequirementOriginV1,
    RequireIndexedEvidencePolicyRuleV1,
    RequireIndexedEvidenceRuleParametersV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import WorkspaceAskRunV2
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    HybridAskPolicyError,
    IndexedEvidenceRequirementV1,
    validate_policy_basis_consistency,
)
from local_workspace_application.workspaces.hybrid_ask_service import (
    WorkspaceAskCommandV2,
    WorkspaceAskServiceV2,
)
from local_workspace_application.workspaces.hybrid_ask_policy_derivation import (
    map_derived_evidence_contract,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveResultRetentionV1,
    QueryPolicyModeV2,
)
from local_workspace_application.tests.workspaces.test_hybrid_ask_service import (
    _Catalog,
    _Configuration,
    _EnvelopeValidator,
    _IndexedRetriever,
    _LiveExecutor,
    _RecordingLLM,
    _Repository,
    _ScopeValidator,
    _WorkspaceAuthority,
    _indexed_evidence,
    _TENANT,
    _WORKSPACE,
)
from intergrax.runtime.vendor_knowledge.live.contracts import KnowledgeQueryAudienceV1
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryOrchestratorV1,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _pentest_rule() -> RequireIndexedEvidencePolicyRuleV1:
    return RequireIndexedEvidencePolicyRuleV1(
        policy_document_id="deployment-policy",
        revision_id="rev18",
        rule_id="RULE-SEC-DEP-4",
        parameters=RequireIndexedEvidenceRuleParametersV1(
            semantic_role="Penetration test evidence",
            requirement_key="pentest",
        ),
    )


def _derive_pentest_contract():
    engine = DeterministicEvidenceObligationDerivation()
    return engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            resolved_policy_rules=(_pentest_rule(),),
        )
    )


def test_pentest_origin_preserved_through_mapping() -> None:
    contract = _derive_pentest_contract()
    _, obligations, policy_basis = map_derived_evidence_contract(contract)

    assert policy_basis is not None
    assert policy_basis.derivation_snapshot_id == contract.derivation_snapshot_id
    obligation = obligations[0]
    assert isinstance(obligation, IndexedEvidenceRequirementV1)
    assert obligation.policy_origin is not None
    assert obligation.policy_origin.policy_document_id == "deployment-policy"
    assert obligation.policy_origin.revision_id == "rev18"
    assert obligation.policy_origin.rule_id == "RULE-SEC-DEP-4"
    assert obligation.requirement_id == (
        "policy:deployment-policy:RULE-SEC-DEP-4:pentest"
    )


def test_policy_basis_required_when_policy_origins_present() -> None:
    origin = RequirementOriginV1(
        policy_document_id="deployment-policy",
        revision_id="rev18",
        rule_id="RULE-SEC-DEP-4",
    )
    obligations = (
        IndexedEvidenceRequirementV1(
            requirement_id="policy:deployment-policy:RULE-SEC-DEP-4:pentest",
            semantic_role="Penetration test evidence",
            policy_origin=origin,
        ),
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        validate_policy_basis_consistency(
            policy_basis=None,
            obligations=obligations,
        )
    assert exc.value.error_code == "policy_basis_missing"


def test_caller_cannot_spoof_policy_origin() -> None:
    with pytest.raises(ValidationError) as exc:
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Why pentest?",
            requested_mode=QueryPolicyModeV2.INDEXED_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.SHARED
            ),
            required_evidence_obligations=(
                IndexedEvidenceRequirementV1(
                    requirement_id="caller:fake-policy",
                    semantic_role="Spoofed policy origin",
                    policy_origin=RequirementOriginV1(
                        policy_document_id="deployment-policy",
                        revision_id="rev18",
                        rule_id="RULE-SEC-DEP-4",
                    ),
                ),
            ),
        )
    assert "caller_policy_origin_forbidden" in str(exc.value)


def test_workspace_ask_run_persists_policy_basis_and_origins() -> None:
    contract = _derive_pentest_contract()
    _, obligations, policy_basis = map_derived_evidence_contract(contract)
    run = WorkspaceAskRunV2(
        run_id="run-provenance-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        question="Why pentest?",
        status=AskRunStatus.FAILED,
        query_mode=QueryPolicyModeV2.INDEXED_ONLY,
        configuration_revision=1,
        plan_id="plan-provenance-1",
        required_evidence_obligations=obligations,
        policy_basis=policy_basis,
        created_at=_NOW,
    )

    assert run.policy_basis is not None
    assert run.policy_basis.policy_revisions == (
        PolicyRevisionReferenceV1(
            policy_document_id="deployment-policy",
            revision_id="rev18",
        ),
    )
    obligation = run.required_evidence_obligations[0]
    assert isinstance(obligation, IndexedEvidenceRequirementV1)
    assert obligation.policy_origin == RequirementOriginV1(
        policy_document_id="deployment-policy",
        revision_id="rev18",
        rule_id="RULE-SEC-DEP-4",
    )


class _FixedPolicyRulesPort:
    def resolve_policy_rules(self, **_: object):
        return (_pentest_rule(),)


def test_service_path_preserves_policy_provenance_on_run() -> None:
    indexed = _IndexedRetriever((_indexed_evidence(),))
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=_LiveExecutor(),  # type: ignore[arg-type]
        clock=lambda: _NOW,
        monotonic=lambda: 100.0,
    )
    repository = _Repository()
    ask_repository = WorkspaceAskRepository(InMemoryDocumentStore())
    configuration = _Configuration(
        QueryPolicyModeV2.INDEXED_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
    )
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=repository,  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=configuration,  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
        request_envelope_validator=_EnvelopeValidator(),  # type: ignore[arg-type]
        resource_scope_validator=_ScopeValidator(),
        orchestrator=orchestrator,
        llm_adapter=_RecordingLLM([_indexed_evidence().evidence_id]),
        clock=lambda: _NOW,
        run_id_factory=lambda: "run-service-provenance",
        plan_id_factory=lambda: "plan-service-provenance",
        evidence_obligation_derivation_port=DeterministicEvidenceObligationDerivation(),
        resolved_policy_rules_port=_FixedPolicyRulesPort(),
    )

    run = asyncio.run(
        service.ask(
            WorkspaceAskCommandV2(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                question="Why pentest?",
                requested_mode=QueryPolicyModeV2.INDEXED_ONLY,
                audience_context=AudienceContextV1(
                    audience=KnowledgeQueryAudienceV1.SHARED
                ),
            )
        )
    )

    assert run.policy_basis is not None
    assert run.policy_basis.policy_revisions == (
        PolicyRevisionReferenceV1(
            policy_document_id="deployment-policy",
            revision_id="rev18",
        ),
    )
    pentest_obligation = next(
        item
        for item in run.required_evidence_obligations
        if item.requirement_id.endswith(":pentest")
    )
    assert isinstance(pentest_obligation, IndexedEvidenceRequirementV1)
    assert pentest_obligation.policy_origin == RequirementOriginV1(
        policy_document_id="deployment-policy",
        revision_id="rev18",
        rule_id="RULE-SEC-DEP-4",
    )
    reloaded = ask_repository.get_run_v2(tenant_id=_TENANT, run_id=run.run_id)
    assert reloaded is not None
    assert reloaded.policy_basis == run.policy_basis
    assert reloaded.required_evidence_obligations == run.required_evidence_obligations
