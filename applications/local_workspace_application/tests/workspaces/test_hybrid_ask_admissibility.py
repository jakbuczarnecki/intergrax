# © Artur Czarnecki. All rights reserved.

"""Evidence Admissibility contract and deterministic evaluator tests."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from pydantic import ValidationError
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
    derive_product_evidence_obligations,
    validate_evidence_plan,
)
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    EvidenceAdmissibilityResultV1,
    EvidenceAdmissibilityStatusV1,
    EvidenceTypeV1,
    IndexedWorkspaceCitationV1,
    IndexedWorkspaceEvidenceV1,
    LiveWorkspaceCitationV1,
    LiveWorkspaceEvidenceV1,
    PersistedIndexedEvidenceV2,
    PersistedLiveEvidenceProvenanceV2,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
    RequiredEvidenceEvaluationV1,
    WorkspaceAskRunV2,
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


def test_product_obligations_do_not_derive_per_planned_live_call() -> None:
    proposals = (
        LiveCallProposalV1(
            call_id="call-required",
            live_access_binding_id="live-1",
            capability_id=_CAP,
            typed_capability_request={"item_key": "ITEM-1"},
        ),
        LiveCallProposalV1(
            call_id="call-optional",
            live_access_binding_id="live-1",
            capability_id=_CAP,
            typed_capability_request={"item_key": "ITEM-2"},
        ),
    )
    obligations = derive_product_evidence_obligations(
        mode=QueryPolicyModeV2.HYBRID,
        include_indexed_retrieval=True,
    )
    assert len(obligations) == 1
    assert obligations[0].requirement_id == "product:hybrid:indexed"
    assert not any(
        isinstance(item, LiveEvidenceRequirementV1) for item in obligations
    )
    del proposals


def test_required_planned_call_satisfied_optional_planned_absent_is_satisfied() -> None:
    obligations = (
        IndexedEvidenceRequirementV1(
            requirement_id="req-indexed",
            semantic_role="Indexed grounding",
        ),
        LiveEvidenceRequirementV1(
            requirement_id="req-live-required",
            semantic_role="Required live call",
            call_id="call-required",
        ),
    )
    result = evaluate_evidence_admissibility(
        obligations=obligations,
        indexed_evidence=(_indexed_evidence(),),
        live_evidence=(_live_evidence(call_id="call-required"),),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED


def test_required_planned_call_absent_optional_present_is_unsatisfied() -> None:
    obligations = (
        IndexedEvidenceRequirementV1(
            requirement_id="req-indexed",
            semantic_role="Indexed grounding",
        ),
        LiveEvidenceRequirementV1(
            requirement_id="req-live-required",
            semantic_role="Required live call",
            call_id="call-required",
        ),
    )
    result = evaluate_evidence_admissibility(
        obligations=obligations,
        indexed_evidence=(_indexed_evidence(),),
        live_evidence=(_live_evidence(call_id="call-optional"),),
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED
    live_eval = next(
        item
        for item in result.requirement_evaluations
        if item.requirement_id == "req-live-required"
    )
    assert live_eval.reason_code is RequirementAdmissibilityReasonCodeV1.LIVE_CALL_MISMATCH


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


def _persisted_indexed(
    *,
    evidence_id: str = "idx:ws-1:doc-1:chunk-1",
    binding_id: str | None = "idx:binding-1",
) -> PersistedIndexedEvidenceV2:
    return PersistedIndexedEvidenceV2(
        evidence_id=evidence_id,
        safe_display_name="document.txt",
        retrieved_at=_NOW,
        content_hash=sha256(b"indexed").hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        source_id="source-1",
        document_id="document-1",
        chunk_id="chunk-1",
        indexed_source_binding_id=binding_id,
    )


def _persisted_live(
    *,
    evidence_id: str = "live:call-1:item-1",
    call_id: str = "call-1",
) -> PersistedLiveEvidenceProvenanceV2:
    return PersistedLiveEvidenceProvenanceV2(
        evidence_id=evidence_id,
        safe_display_name="Live item",
        retrieved_at=_NOW,
        content_hash=sha256(b"live").hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        provider_id=_PROVIDER,
        live_access_binding_id="live-1",
        connection_ref="conn.live",
        capability_id=_CAP,
        call_id=call_id,
    )


def _valid_satisfied_run_payload() -> dict[str, object]:
    indexed = _persisted_indexed()
    live = _persisted_live()
    obligations = (
        IndexedEvidenceRequirementV1(
            requirement_id="req-indexed",
            semantic_role="Indexed grounding",
            indexed_source_binding_id=_INDEXED_BINDING,
        ),
        LiveEvidenceRequirementV1(
            requirement_id="req-live",
            semantic_role="Live grounding",
            call_id="call-1",
        ),
    )
    admissibility = EvidenceAdmissibilityResultV1(
        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
        requirement_evaluations=(
            RequiredEvidenceEvaluationV1(
                requirement_id="req-indexed",
                status=RequirementEvaluationStatusV1.SATISFIED,
                matched_evidence_ids=(indexed.evidence_id,),
            ),
            RequiredEvidenceEvaluationV1(
                requirement_id="req-live",
                status=RequirementEvaluationStatusV1.SATISFIED,
                matched_evidence_ids=(live.evidence_id,),
            ),
        ),
    )
    return {
        "run_id": "run-valid",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "question": "Valid admissibility record?",
        "status": AskRunStatus.COMPLETED,
        "query_mode": QueryPolicyModeV2.HYBRID,
        "configuration_revision": 1,
        "plan_id": "plan-valid",
        "required_evidence_obligations": obligations,
        "evidence_admissibility": admissibility,
        "created_at": _NOW,
        "completed_at": _NOW,
        "persisted_evidence": [indexed, live],
        "citations": [
            IndexedWorkspaceCitationV1(
                evidence_id=indexed.evidence_id,
                safe_display_name=indexed.safe_display_name,
                retrieved_at=indexed.retrieved_at,
                document_id=indexed.document_id,
                source_id=indexed.source_id,
                workspace_id=_WORKSPACE,
                source_path="/docs/document.txt",
                file_name="document.txt",
            ),
            LiveWorkspaceCitationV1(
                evidence_id=live.evidence_id,
                safe_display_name=live.safe_display_name,
                retrieved_at=live.retrieved_at,
                provider_id=live.provider_id,
                connection_safe_label="Live connection",
                capability_id=live.capability_id,
                call_id=live.call_id,
            ),
        ],
    }


def _valid_unsatisfied_run_payload() -> dict[str, object]:
    obligations = (
        IndexedEvidenceRequirementV1(
            requirement_id="req-indexed",
            semantic_role="Indexed grounding",
        ),
    )
    admissibility = EvidenceAdmissibilityResultV1(
        overall_status=EvidenceAdmissibilityStatusV1.UNSATISFIED,
        requirement_evaluations=(
            RequiredEvidenceEvaluationV1(
                requirement_id="req-indexed",
                status=RequirementEvaluationStatusV1.UNSATISFIED,
                reason_code=RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE,
            ),
        ),
    )
    return {
        "run_id": "run-insufficient",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "question": "Missing indexed evidence?",
        "status": AskRunStatus.INSUFFICIENT_EVIDENCE,
        "query_mode": QueryPolicyModeV2.INDEXED_ONLY,
        "configuration_revision": 1,
        "plan_id": "plan-insufficient",
        "required_evidence_obligations": obligations,
        "evidence_admissibility": admissibility,
        "created_at": _NOW,
        "completed_at": _NOW,
        "persisted_evidence": [],
        "citations": [],
    }


def test_valid_satisfied_admissibility_run_reconstructs() -> None:
    run = WorkspaceAskRunV2.model_validate(_valid_satisfied_run_payload())
    assert run.evidence_admissibility is not None
    assert run.evidence_admissibility.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED


def test_valid_unsatisfied_admissibility_run_reconstructs() -> None:
    run = WorkspaceAskRunV2.model_validate(_valid_unsatisfied_run_payload())
    assert run.evidence_admissibility is not None
    assert run.evidence_admissibility.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": payload["evidence_admissibility"].model_copy(
                        update={
                            "requirement_evaluations": (
                                RequiredEvidenceEvaluationV1(
                                    requirement_id="req-unknown",
                                    status=RequirementEvaluationStatusV1.UNSATISFIED,
                                    reason_code=RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE,
                                ),
                            )
                        }
                    )
                }
            ),
            "admissibility_unknown_requirement_id",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                            ),
                        ),
                    )
                }
            ),
            "duplicate_admissibility_requirement_id",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=(),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                        ),
                    )
                }
            ),
            "satisfied_evaluation_requires_matched_evidence",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                                reason_code=RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE,
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                        ),
                    )
                }
            ),
            "satisfied_evaluation_forbids_reason_code",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.UNSATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.UNSATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                                reason_code=RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE,
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                        ),
                    )
                }
            ),
            "unsatisfied_evaluation_forbids_matched_evidence",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:missing",),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                        ),
                    )
                }
            ),
            "matched_evidence_not_persisted",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                        ),
                    )
                }
            ),
            "indexed_obligation_evidence_type_mismatch",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                            ),
                        ),
                    )
                }
            ),
            "live_obligation_evidence_type_mismatch",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-2:item-1",),
                            ),
                        ),
                    ),
                    "persisted_evidence": [
                        _persisted_indexed(),
                        _persisted_live(evidence_id="live:call-2:item-1", call_id="call-2"),
                    ],
                    "citations": [
                        IndexedWorkspaceCitationV1(
                            evidence_id="idx:ws-1:doc-1:chunk-1",
                            safe_display_name="document.txt",
                            retrieved_at=_NOW,
                            document_id="document-1",
                            source_id="source-1",
                            workspace_id=_WORKSPACE,
                            source_path="/docs/document.txt",
                            file_name="document.txt",
                        ),
                        LiveWorkspaceCitationV1(
                            evidence_id="live:call-2:item-1",
                            safe_display_name="Live item",
                            retrieved_at=_NOW,
                            provider_id=_PROVIDER,
                            connection_safe_label="Live connection",
                            capability_id=_CAP,
                            call_id="call-2",
                        ),
                    ],
                }
            ),
            "live_obligation_call_id_mismatch",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:other-binding",),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                        ),
                    ),
                    "persisted_evidence": [
                        _persisted_indexed(
                            evidence_id="idx:other-binding",
                            binding_id=_OTHER_BINDING,
                        ),
                        _persisted_live(),
                    ],
                    "citations": [
                        IndexedWorkspaceCitationV1(
                            evidence_id="idx:other-binding",
                            safe_display_name="document.txt",
                            retrieved_at=_NOW,
                            document_id="document-1",
                            source_id="source-1",
                            workspace_id=_WORKSPACE,
                            source_path="/docs/document.txt",
                            file_name="document.txt",
                        ),
                        LiveWorkspaceCitationV1(
                            evidence_id="live:call-1:item-1",
                            safe_display_name="Live item",
                            retrieved_at=_NOW,
                            provider_id=_PROVIDER,
                            connection_safe_label="Live connection",
                            capability_id=_CAP,
                            call_id="call-1",
                        ),
                    ],
                }
            ),
            "indexed_obligation_binding_mismatch",
        ),
        (
            lambda payload: payload.update(
                {
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("idx:ws-1:doc-1:chunk-1",),
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.UNSATISFIED,
                                reason_code=RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE,
                            ),
                        ),
                    )
                }
            ),
            "admissibility_overall_status_mismatch",
        ),
        (
            lambda payload: payload.update(
                {
                    "status": AskRunStatus.COMPLETED,
                    "evidence_admissibility": EvidenceAdmissibilityResultV1(
                        overall_status=EvidenceAdmissibilityStatusV1.UNSATISFIED,
                        requirement_evaluations=(
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-indexed",
                                status=RequirementEvaluationStatusV1.UNSATISFIED,
                                reason_code=RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE,
                            ),
                            RequiredEvidenceEvaluationV1(
                                requirement_id="req-live",
                                status=RequirementEvaluationStatusV1.SATISFIED,
                                matched_evidence_ids=("live:call-1:item-1",),
                            ),
                        ),
                    ),
                }
            ),
            "completed_run_requires_satisfied_admissibility",
        ),
    ],
)
def test_run_integrity_rejects_tampered_admissibility(
    mutator: object,
    expected_error: str,
) -> None:
    payload = _valid_satisfied_run_payload()
    mutator(payload)
    with pytest.raises(ValidationError) as exc:
        WorkspaceAskRunV2.model_validate(payload)
    assert expected_error in str(exc.value)
