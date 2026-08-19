# © Artur Czarnecki. All rights reserved.

"""Evidence Admissibility contract and deterministic evaluator tests."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from local_workspace_application.workspaces.hybrid_ask_admissibility import (
    evaluate_evidence_admissibility,
    evaluate_execution_admissibility,
)
from local_workspace_application.workspaces.hybrid_ask_execution import (
    HybridAskIndexedRetrievalStatusV1,
    HybridAskLiveExecutionStatusV1,
    HybridAskTruncationStateV1,
    KnowledgeQueryExecutionResultV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    EvidenceAdmissibilityStatusV1,
    IndexedWorkspaceEvidenceV1,
    LiveWorkspaceEvidenceV1,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    EvidencePlanV1,
    HybridAskPolicyError,
    IndexedEvidenceRequirementV1,
    IndexedRetrievalDirectiveV1,
    KnowledgeQueryAudienceV1,
    LiveCallProposalV1,
    LiveEvidenceRequirementV1,
    ValidatedEvidencePlanV1,
    validate_evidence_plan,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    QueryPolicyModeV2,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.contracts import EffectiveLiveCallBudgetV1

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "tenant-1"
_WORKSPACE = "workspace-1"
_PROVIDER = "neutral_provider"
_KIND = IntegrationCategory.WIKI_KNOWLEDGE
_CAP = "vendor.neutral_provider.issues.read"
_INDEXED_BINDING = "idx:binding-1"
_OTHER_BINDING = "idx:binding-2"


def _budget() -> EffectiveLiveCallBudgetV1:
    return EffectiveLiveCallBudgetV1(
        max_live_calls=2,
        max_total_duration_ms=30_000,
        max_result_items=50,
        max_result_bytes=1_048_576,
    )


def _indexed_evidence(
    *,
    binding_id: str | None = None,
    evidence_id: str = "idx:ws-1:doc-1:chunk-1",
) -> IndexedWorkspaceEvidenceV1:
    content = "Indexed content."
    return IndexedWorkspaceEvidenceV1(
        evidence_id=evidence_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        safe_display_name="document.txt",
        retrieved_at=_NOW,
        content=content,
        content_hash=sha256(content.encode()).hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        source_id="source-1",
        document_id="document-1",
        chunk_id="chunk-1",
        indexed_source_binding_id=binding_id,
    )


def _live_evidence(*, call_id: str = "call-1") -> LiveWorkspaceEvidenceV1:
    content = "Live content."
    return LiveWorkspaceEvidenceV1(
        evidence_id=f"live:{call_id}:item-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        safe_display_name="Live item",
        retrieved_at=_NOW,
        content=content,
        content_hash=sha256(content.encode()).hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        live_access_binding_id="live-1",
        connection_ref="conn.live",
        capability_id=_CAP,
        source_kind="issues",
        contract_version="1",
        provider_id=_PROVIDER,
        integration_kind=_KIND.value,
        call_id=call_id,
    )


def _execution(
    *,
    indexed: tuple[IndexedWorkspaceEvidenceV1, ...] = (),
    live: tuple[LiveWorkspaceEvidenceV1, ...] = (),
) -> KnowledgeQueryExecutionResultV1:
    return KnowledgeQueryExecutionResultV1(
        run_id="run-1",
        plan_id="plan-1",
        mode=QueryPolicyModeV2.HYBRID,
        indexed_evidence=indexed,
        live_evidence=live,
        receipts=(),
        indexed_retrieval_status=HybridAskIndexedRetrievalStatusV1.COMPLETED,
        live_execution_status=HybridAskLiveExecutionStatusV1.COMPLETED,
        truncation_state=HybridAskTruncationStateV1.NONE,
        partial_failure=False,
        started_at=_NOW,
        completed_at=_NOW,
    )


def test_admissibility_empty_obligations_is_satisfied() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(),
        indexed_evidence=(_indexed_evidence(),),
        live_evidence=(),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED
    assert result.requirement_evaluations == ()


def test_admissibility_all_required_evidence_satisfied() -> None:
    obligations = (
        IndexedEvidenceRequirementV1(
            requirement_id="req-indexed",
            semantic_role="Grounding corpus",
        ),
        LiveEvidenceRequirementV1(
            requirement_id="req-live",
            semantic_role="Authoritative live state",
            call_id="call-1",
        ),
    )
    indexed = _indexed_evidence()
    live = _live_evidence(call_id="call-1")
    result = evaluate_evidence_admissibility(
        obligations=obligations,
        indexed_evidence=(indexed,),
        live_evidence=(live,),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED
    assert len(result.requirement_evaluations) == 2
    assert all(
        item.status is RequirementEvaluationStatusV1.SATISFIED
        for item in result.requirement_evaluations
    )


def test_admissibility_missing_indexed_evidence_is_unsatisfied() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(
            IndexedEvidenceRequirementV1(
                requirement_id="req-indexed",
                semantic_role="Grounding corpus",
            ),
        ),
        indexed_evidence=(),
        live_evidence=(),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED
    evaluation = result.requirement_evaluations[0]
    assert evaluation.reason_code is RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE


def test_admissibility_missing_live_evidence_is_unsatisfied() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(
            LiveEvidenceRequirementV1(
                requirement_id="req-live",
                semantic_role="Authoritative live state",
                call_id="call-1",
            ),
        ),
        indexed_evidence=(),
        live_evidence=(),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED


def test_admissibility_wrong_live_call_cannot_satisfy_requirement() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(
            LiveEvidenceRequirementV1(
                requirement_id="req-live-a",
                semantic_role="Call A evidence",
                call_id="call-a",
            ),
        ),
        indexed_evidence=(),
        live_evidence=(_live_evidence(call_id="call-b"),),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED
    assert (
        result.requirement_evaluations[0].reason_code
        is RequirementAdmissibilityReasonCodeV1.LIVE_CALL_MISMATCH
    )


def test_admissibility_wrong_indexed_binding_cannot_satisfy_scoped_requirement() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(
            IndexedEvidenceRequirementV1(
                requirement_id="req-indexed",
                semantic_role="Specific binding",
                indexed_source_binding_id=_INDEXED_BINDING,
            ),
        ),
        indexed_evidence=(_indexed_evidence(binding_id=_OTHER_BINDING),),
        live_evidence=(),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED
    assert (
        result.requirement_evaluations[0].reason_code
        is RequirementAdmissibilityReasonCodeV1.INDEXED_BINDING_MISMATCH
    )


def test_admissibility_scoped_indexed_binding_satisfied() -> None:
    indexed = _indexed_evidence(binding_id=_INDEXED_BINDING)
    result = evaluate_evidence_admissibility(
        obligations=(
            IndexedEvidenceRequirementV1(
                requirement_id="req-indexed",
                semantic_role="Specific binding",
                indexed_source_binding_id=_INDEXED_BINDING,
            ),
        ),
        indexed_evidence=(indexed,),
        live_evidence=(),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED
    assert result.requirement_evaluations[0].matched_evidence_ids == (indexed.evidence_id,)


def test_execution_admissibility_is_deterministic_without_side_effects() -> None:
    plan = ValidatedEvidencePlanV1(
        plan=EvidencePlanV1(
            plan_id="plan-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            mode=QueryPolicyModeV2.HYBRID,
            indexed_retrieval_directive=IndexedRetrievalDirectiveV1(max_results=5),
            ordered_live_call_proposals=(
                LiveCallProposalV1(
                    call_id="call-1",
                    live_access_binding_id="live-1",
                    capability_id=_CAP,
                    typed_capability_request={"item_key": "ITEM-1"},
                ),
            ),
            required_evidence_obligations=(
                LiveEvidenceRequirementV1(
                    requirement_id="req-live",
                    semantic_role="Live proof",
                    call_id="call-1",
                ),
            ),
            budget_snapshot=_budget(),
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
        ),
        executable_live_calls=(),
        effective_budget=_budget(),
    )
    execution = _execution(live=(_live_evidence(call_id="call-1"),))
    first = evaluate_execution_admissibility(
        validated_plan=plan,
        execution=execution,
    )
    second = evaluate_execution_admissibility(
        validated_plan=plan,
        execution=execution,
    )
    assert first == second


def test_plan_validation_rejects_duplicate_requirement_id() -> None:
    from local_workspace_application.tests.workspaces.test_hybrid_ask_core import (
        _FakeCatalog,
        _FakeEnvelopeValidator,
        _FakeScopeValidator,
        _configuration,
        _descriptor,
        _hybrid_plan,
        _v2_policy,
        resolve_effective_query_policy,
    )

    config = _configuration(
        revision=1,
        query_policy=_v2_policy(mode=QueryPolicyModeV2.HYBRID, effective_revision=1),
    )
    effective = resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.HYBRID,
        configuration=config,
        configuration_revision=1,
    )
    plan = _hybrid_plan().model_copy(
        update={
            "required_evidence_obligations": (
                IndexedEvidenceRequirementV1(
                    requirement_id="dup",
                    semantic_role="First",
                ),
                IndexedEvidenceRequirementV1(
                    requirement_id="dup",
                    semantic_role="Second",
                ),
            )
        }
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        validate_evidence_plan(
            plan=plan,
            configuration=config,
            effective_policy=effective,
            capability_catalog=_FakeCatalog({"conn.live": _descriptor()}),
            request_envelope_validator=_FakeEnvelopeValidator(),
            resource_scope_validator=_FakeScopeValidator(),
        )
    assert exc.value.error_code == "duplicate_requirement_id"


def test_plan_validation_rejects_unknown_live_call_reference() -> None:
    from local_workspace_application.tests.workspaces.test_hybrid_ask_core import (
        _FakeCatalog,
        _FakeEnvelopeValidator,
        _FakeScopeValidator,
        _configuration,
        _descriptor,
        _hybrid_plan,
        _v2_policy,
        resolve_effective_query_policy,
    )

    config = _configuration(
        revision=1,
        query_policy=_v2_policy(mode=QueryPolicyModeV2.HYBRID, effective_revision=1),
    )
    effective = resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.HYBRID,
        configuration=config,
        configuration_revision=1,
    )
    plan = _hybrid_plan().model_copy(
        update={
            "required_evidence_obligations": (
                LiveEvidenceRequirementV1(
                    requirement_id="req-live",
                    semantic_role="Missing call",
                    call_id="call-unknown",
                ),
            )
        }
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        validate_evidence_plan(
            plan=plan,
            configuration=config,
            effective_policy=effective,
            capability_catalog=_FakeCatalog({"conn.live": _descriptor()}),
            request_envelope_validator=_FakeEnvelopeValidator(),
            resource_scope_validator=_FakeScopeValidator(),
        )
    assert exc.value.error_code == "unknown_live_call_reference"
