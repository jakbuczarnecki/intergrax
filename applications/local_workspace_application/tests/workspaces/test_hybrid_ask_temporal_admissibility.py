# © Artur Czarnecki. All rights reserved.

"""Temporal evidence admissibility tests (COMM-5F3-D)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
    derive_derivation_snapshot_id,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    EvidenceObligationDerivationContextV1,
    MaxAgeTemporalConstraintV1,
    PointInTimeEvidenceTemporalV1,
    RequireLiveEvidencePolicyRuleV1,
    RequireLiveEvidenceRuleParametersV1,
    TypedCapabilityRequestEntryV1,
    ValidAtTemporalConstraintV1,
    ValidityIntervalEvidenceTemporalV1,
)
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_admissibility import (
    evaluate_evidence_admissibility,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    EvidenceAdmissibilityStatusV1,
    EvidenceTypeV1,
    LiveWorkspaceEvidenceV1,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    LiveEvidenceRequirementV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy_derivation import (
    map_derived_obligation,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    QueryPolicyModeV2,
)

pytestmark = pytest.mark.unit

_EVALUATED_AT = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
_TENANT = "tenant-temporal"
_WORKSPACE = "workspace-temporal"
_MAX_AGE = 3600
_SECURITY_REQ = "policy:security-policy:RULE-SECURITY:security"
_SECURITY_CALL = "policy-call:security-policy:RULE-SECURITY:security-read"


def _live_evidence(
    *,
    effective_at: datetime,
    call_id: str = _SECURITY_CALL,
    evidence_id: str = "live:security:item-1",
) -> LiveWorkspaceEvidenceV1:
    content = '{"status":"clear"}'
    return LiveWorkspaceEvidenceV1(
        evidence_id=evidence_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        safe_display_name="Security status",
        retrieved_at=_EVALUATED_AT,
        content=content,
        content_hash=sha256(content.encode()).hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        live_access_binding_id="binding-security",
        connection_ref="conn.security-status",
        capability_id="vendor.security_status.security.read",
        source_kind="security",
        contract_version="1",
        provider_id="security_status",
        integration_kind="security_scanner",
        call_id=call_id,
        temporal=PointInTimeEvidenceTemporalV1(effective_at=effective_at),
    )


def _security_requirement(
    *,
    max_age_seconds: int | None = _MAX_AGE,
) -> LiveEvidenceRequirementV1:
    return LiveEvidenceRequirementV1(
        requirement_id=_SECURITY_REQ,
        semantic_role="Security blocker status",
        call_id=_SECURITY_CALL,
        temporal_constraint=(
            None
            if max_age_seconds is None
            else MaxAgeTemporalConstraintV1(max_age_seconds=max_age_seconds)
        ),
    )


def test_fresh_evidence_is_satisfied() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(_security_requirement(),),
        indexed_evidence=(),
        live_evidence=(
            _live_evidence(
                effective_at=_EVALUATED_AT - timedelta(minutes=30),
            ),
        ),
        evaluated_at=_EVALUATED_AT,
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED
    assert result.evaluated_at == _EVALUATED_AT


def test_stale_evidence_is_unsatisfied_and_suppresses_llm_path() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(_security_requirement(),),
        indexed_evidence=(),
        live_evidence=(
            _live_evidence(
                effective_at=_EVALUATED_AT - timedelta(hours=2),
            ),
        ),
        evaluated_at=_EVALUATED_AT,
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED
    evaluation = result.requirement_evaluations[0]
    assert evaluation.status is RequirementEvaluationStatusV1.UNSATISFIED
    assert (
        evaluation.reason_code
        is RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID
    )
    assert evaluation.matched_evidence_ids == ()


def test_exact_max_age_boundary_is_valid() -> None:
    exact = evaluate_evidence_admissibility(
        obligations=(_security_requirement(),),
        indexed_evidence=(),
        live_evidence=(
            _live_evidence(
                effective_at=_EVALUATED_AT - timedelta(seconds=_MAX_AGE),
            ),
        ),
        evaluated_at=_EVALUATED_AT,
    )
    assert exact.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED

    beyond = evaluate_evidence_admissibility(
        obligations=(_security_requirement(),),
        indexed_evidence=(),
        live_evidence=(
            _live_evidence(
                effective_at=_EVALUATED_AT - timedelta(seconds=_MAX_AGE, microseconds=1),
            ),
        ),
        evaluated_at=_EVALUATED_AT,
    )
    assert beyond.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED


def test_future_dated_point_evidence_fails_closed() -> None:
    result = evaluate_evidence_admissibility(
        obligations=(_security_requirement(),),
        indexed_evidence=(),
        live_evidence=(
            _live_evidence(
                effective_at=_EVALUATED_AT + timedelta(seconds=1),
            ),
        ),
        evaluated_at=_EVALUATED_AT,
    )
    assert result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED
    assert (
        result.requirement_evaluations[0].reason_code
        is RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID
    )


def test_validity_interval_boundaries() -> None:
    interval_from = datetime(2026, 8, 20, 17, 0, tzinfo=UTC)
    interval_until = datetime(2026, 8, 20, 19, 0, tzinfo=UTC)
    requirement = LiveEvidenceRequirementV1(
        requirement_id="req-interval",
        semantic_role="Approval validity",
        call_id="call-interval",
        temporal_constraint=ValidAtTemporalConstraintV1(),
    )
    evidence = _live_evidence(
        effective_at=interval_from,
        call_id="call-interval",
        evidence_id="live:interval:item-1",
    )
    evidence = evidence.model_copy(
        update={
            "temporal": ValidityIntervalEvidenceTemporalV1(
                valid_from=interval_from,
                valid_until=interval_until,
            )
        }
    )
    cases = (
        (datetime(2026, 8, 20, 16, 59, tzinfo=UTC), False),
        (interval_from, True),
        (datetime(2026, 8, 20, 18, 0, tzinfo=UTC), True),
        (interval_until, True),
        (datetime(2026, 8, 20, 19, 1, tzinfo=UTC), False),
    )
    for evaluated_at, expected_satisfied in cases:
        result = evaluate_evidence_admissibility(
            obligations=(requirement,),
            indexed_evidence=(),
            live_evidence=(evidence,),
            evaluated_at=evaluated_at,
        )
        expected = (
            EvidenceAdmissibilityStatusV1.SATISFIED
            if expected_satisfied
            else EvidenceAdmissibilityStatusV1.UNSATISFIED
        )
        assert result.overall_status is expected, evaluated_at.isoformat()


def _security_live_rule(
    *,
    revision_id: str,
    max_age_seconds: int | None,
) -> RequireLiveEvidencePolicyRuleV1:
    return RequireLiveEvidencePolicyRuleV1(
        policy_document_id="security-policy",
        revision_id=revision_id,
        rule_id="RULE-SECURITY",
        parameters=RequireLiveEvidenceRuleParametersV1(
            semantic_role="Security blocker status",
            requirement_key="security",
            capability_id="vendor.security_status.security.read",
            live_access_binding_id="binding-security",
            live_call_descriptor_ref="security-read",
            typed_capability_request=(
                TypedCapabilityRequestEntryV1(key="project_id", value="ORION-1"),
            ),
            temporal_constraint=(
                None
                if max_age_seconds is None
                else MaxAgeTemporalConstraintV1(max_age_seconds=max_age_seconds)
            ),
        ),
    )


def test_revision_changes_temporal_policy_without_requirement_id_change() -> None:
    rev17_rule = _security_live_rule(revision_id="17", max_age_seconds=86_400)
    rev18_rule = _security_live_rule(revision_id="18", max_age_seconds=3_600)
    engine = DeterministicEvidenceObligationDerivation()
    rev17 = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=17,
            resolved_policy_rules=(rev17_rule,),
        )
    )
    rev18 = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=18,
            resolved_policy_rules=(rev18_rule,),
        )
    )
    rev17_obligation = map_derived_obligation(rev17.derived_obligations[0])
    rev18_obligation = map_derived_obligation(rev18.derived_obligations[0])
    assert rev17_obligation.requirement_id == rev18_obligation.requirement_id
    assert rev17.derivation_snapshot_id != rev18.derivation_snapshot_id

    evidence = _live_evidence(
        effective_at=_EVALUATED_AT - timedelta(hours=2),
    )
    rev17_result = evaluate_evidence_admissibility(
        obligations=(rev17_obligation,),
        indexed_evidence=(),
        live_evidence=(evidence,),
        evaluated_at=_EVALUATED_AT,
    )
    rev18_result = evaluate_evidence_admissibility(
        obligations=(rev18_obligation,),
        indexed_evidence=(),
        live_evidence=(evidence,),
        evaluated_at=_EVALUATED_AT,
    )
    assert rev17_result.overall_status is EvidenceAdmissibilityStatusV1.SATISFIED
    assert rev18_result.overall_status is EvidenceAdmissibilityStatusV1.UNSATISFIED


def test_derivation_snapshot_changes_when_temporal_rule_changes() -> None:
    rev17 = _security_live_rule(revision_id="17", max_age_seconds=86_400)
    rev18 = _security_live_rule(revision_id="17", max_age_seconds=3_600)
    snapshot17 = derive_derivation_snapshot_id(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=17,
        resolved_policy_rules=(rev17,),
    )
    snapshot18 = derive_derivation_snapshot_id(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=17,
        resolved_policy_rules=(rev18,),
    )
    assert snapshot17 != snapshot18


def test_run_persistence_preserves_temporal_proof() -> None:
    from intergrax.integrations._shared.in_memory_document_store import (
        InMemoryDocumentStore,
    )
    from local_workspace_application.workspaces.hybrid_ask_models import (
        EvidenceAdmissibilityResultV1,
        PersistedLiveEvidenceProvenanceV2,
        RequiredEvidenceEvaluationV1,
        WorkspaceAskRunV2,
    )

    store = InMemoryDocumentStore()
    repository = WorkspaceAskRepository(store)
    obligation = _security_requirement()
    evidence = _live_evidence(
        effective_at=_EVALUATED_AT - timedelta(minutes=30),
    )
    admissibility = evaluate_evidence_admissibility(
        obligations=(obligation,),
        indexed_evidence=(),
        live_evidence=(evidence,),
        evaluated_at=_EVALUATED_AT,
    )
    run = WorkspaceAskRunV2(
        run_id="run-temporal-persist",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        question="Temporal persistence?",
        status=AskRunStatus.INSUFFICIENT_EVIDENCE,
        query_mode=QueryPolicyModeV2.LIVE_ONLY,
        configuration_revision=17,
        plan_id="plan-temporal",
        required_evidence_obligations=(obligation,),
        evidence_admissibility=admissibility,
        persisted_evidence=[
            PersistedLiveEvidenceProvenanceV2(
                evidence_id=evidence.evidence_id,
                safe_display_name=evidence.safe_display_name,
                retrieved_at=evidence.retrieved_at,
                content_hash=evidence.content_hash,
                audience=evidence.audience,
                provider_id=evidence.provider_id,
                live_access_binding_id=evidence.live_access_binding_id,
                connection_ref=evidence.connection_ref,
                capability_id=evidence.capability_id,
                call_id=evidence.call_id,
                temporal=evidence.temporal,
            )
        ],
        created_at=_EVALUATED_AT,
        completed_at=_EVALUATED_AT,
    )
    repository.put_run_v2(run)
    reloaded = repository.get_run_v2(
        tenant_id=_TENANT,
        run_id="run-temporal-persist",
    )
    assert reloaded is not None
    assert reloaded.required_evidence_obligations[0].temporal_constraint is not None
    assert reloaded.evidence_admissibility is not None
    assert reloaded.evidence_admissibility.evaluated_at == _EVALUATED_AT
    live_persisted = next(
        item
        for item in reloaded.persisted_evidence
        if item.evidence_type is EvidenceTypeV1.LIVE
    )
    assert live_persisted.temporal == evidence.temporal
