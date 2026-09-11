# © Artur Czarnecki. All rights reserved.

"""Typed completion-alignment revision context tests (DS-E2E-15I)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    CompletionAlignmentMismatchReason,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_revision_context import (
    CompletionAlignmentRevisionContext,
    build_completion_alignment_revision_context,
    render_completion_alignment_revision_context,
    resolve_authoritative_supported_hypothesis,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    HypothesisId,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    ClaimHypothesisBinding,
    ClaimProposal,
    CompletionIntent,
    HypothesisDisposition,
    HypothesisProposal,
    IncidentReasoningProposal,
    PriorInvestigationState,
    build_reasoning_messages,
)
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_phase_context import (
    EvidencePhaseContext,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    COMPARISON_EVIDENCE_ID,
    DIAGNOSIS_KIND,
    H3_CLAIM_ID,
    INITIAL_CLAIM_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    build_resolved_fixture,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    apply_critic_claim_resolutions,
)

pytestmark = pytest.mark.unit

_H3_BINDING = (
    ClaimHypothesisBinding(claim_id=str(H3_CLAIM_ID), hypothesis_id="H3"),
)


def _resolved_h3_domain_payload() -> dict[str, object]:
    fixture = build_resolved_fixture()
    return {
        "evidence_nodes": [
            {
                "evidence_id": str(WORKLOAD_EVIDENCE_ID),
                "payload": {
                    "order_volume_delta_pct": fixture.workload_incident.order_volume_delta_pct,
                    "admissible": True,
                },
            },
            {
                "evidence_id": str(THROUGHPUT_EVIDENCE_ID),
                "payload": {
                    "target_attainment_pct": fixture.throughput_incident.target_attainment_pct,
                    "baseline_attainment_pct": fixture.throughput_incident.baseline_attainment_pct,
                    "admissible": True,
                },
            },
            {
                "evidence_id": str(STAFFING_PRELIMINARY_EVIDENCE_ID),
                "payload": {
                    "scheduled_headcount": fixture.staffing_preliminary.scheduled_headcount,
                    "required_headcount": fixture.staffing_preliminary.required_headcount,
                    "record_valid_from": (
                        fixture.staffing_preliminary.record_valid_for.observed_from.isoformat()
                    ),
                    "record_valid_to": (
                        fixture.staffing_preliminary.record_valid_for.observed_to.isoformat()
                    ),
                    "window_observed_from": (
                        fixture.staffing_preliminary.window.observed_from.isoformat()
                    ),
                    "window_observed_to": (
                        fixture.staffing_preliminary.window.observed_to.isoformat()
                    ),
                },
            },
            {
                "evidence_id": str(STAFFING_ATTENDANCE_EVIDENCE_ID),
                "payload": {
                    "confirmed_headcount": fixture.staffing_attendance.confirmed_headcount,
                },
            },
            {
                "evidence_id": str(COMPARISON_EVIDENCE_ID),
                "payload": {
                    "workload_delta_pct": fixture.comparison.workload_delta_pct,
                    "comparison_attainment_pct": fixture.comparison.target_attainment_pct,
                    "reference_attainment_pct": fixture.comparison.reference_attainment_pct,
                    "admissible": True,
                },
            },
            {
                "evidence_id": str(TELEMETRY_EVIDENCE_ID),
                "payload": {
                    "availability": "available",
                    "signal_state": fixture.telemetry.signal_state,
                    "complex_assembly_throughput_pct": (
                        fixture.telemetry.complex_assembly_throughput_pct
                    ),
                    "baseline_throughput_pct": fixture.telemetry.baseline_throughput_pct,
                    "admissible": True,
                },
            },
        ],
        "claim_hypothesis_bindings": [
            {"claim_id": str(H3_CLAIM_ID), "hypothesis_id": "H3"},
        ],
    }


def _pending_h3_claim_set() -> EvidenceClaimSet:
    return EvidenceClaimSet(
        claims=(
            EvidenceBackedClaim(
                claim_id=H3_CLAIM_ID,
                statement="Bounded H3 diagnosis.",
                claim_kind=DIAGNOSIS_KIND,
                supporting_evidence_ids=(TELEMETRY_EVIDENCE_ID, COMPARISON_EVIDENCE_ID),
                resolution=ClaimResolution.PENDING,
            ),
        ),
        challenges=(),
    )


def _authoritative_h3_resolved_claim_set() -> EvidenceClaimSet:
    return EvidenceClaimSet(
        claims=(
            EvidenceBackedClaim(
                claim_id=H3_CLAIM_ID,
                statement="Bounded H3 diagnosis.",
                claim_kind=DIAGNOSIS_KIND,
                supporting_evidence_ids=(TELEMETRY_EVIDENCE_ID, COMPARISON_EVIDENCE_ID),
                resolution=ClaimResolution.SUPPORTED,
            ),
        ),
        challenges=(),
    )


def _critic_resolved_h3_claim_set() -> EvidenceClaimSet:
    pending = _pending_h3_claim_set()
    return apply_critic_claim_resolutions(
        pending,
        _resolved_h3_domain_payload(),
        bindings=_H3_BINDING,
    )


def test_revision_context_contract_is_immutable() -> None:
    context = CompletionAlignmentRevisionContext(
        mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        supported_hypothesis_id=HypothesisId.H3,
        supported_resolution=ClaimResolution.SUPPORTED,
    )
    with pytest.raises(FrozenInstanceError):
        context.supported_hypothesis_id = HypothesisId.H1  # type: ignore[misc]


def test_revision_context_valid_target_case() -> None:
    CompletionAlignmentRevisionContext(
        mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        supported_hypothesis_id=HypothesisId.H3,
        supported_resolution=ClaimResolution.SUPPORTED,
    )


def test_revision_context_rejects_target_without_hypothesis() -> None:
    with pytest.raises(ValueError, match="requires supported_hypothesis_id"):
        CompletionAlignmentRevisionContext(
            mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
            supported_hypothesis_id=None,
            supported_resolution=ClaimResolution.SUPPORTED,
        )


def test_revision_context_valid_reverse_case() -> None:
    CompletionAlignmentRevisionContext(
        mismatch_reason=(
            CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
        ),
        supported_hypothesis_id=None,
        supported_resolution=None,
    )


def test_revision_context_rejects_reverse_with_hypothesis() -> None:
    with pytest.raises(ValueError, match="forbids supported_hypothesis_id"):
        CompletionAlignmentRevisionContext(
            mismatch_reason=(
                CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
            ),
            supported_hypothesis_id=HypothesisId.H3,
            supported_resolution=None,
        )


def test_builder_target_case_projects_h3_supported() -> None:
    context = build_completion_alignment_revision_context(
        resolved_claim_set=_authoritative_h3_resolved_claim_set(),
        bindings=_H3_BINDING,
        prior_completion_mode=COMPLETION_UNRESOLVED,
    )
    assert context is not None
    assert context.mismatch_reason is (
        CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS
    )
    assert context.supported_hypothesis_id is HypothesisId.H3
    assert context.supported_resolution is ClaimResolution.SUPPORTED


def test_builder_ignores_pending_model_claims_without_critic_resolution() -> None:
    context = build_completion_alignment_revision_context(
        resolved_claim_set=_pending_h3_claim_set(),
        bindings=_H3_BINDING,
        prior_completion_mode=COMPLETION_UNRESOLVED,
    )
    assert context is None


def test_builder_reverse_mismatch_without_supported_hypothesis() -> None:
    empty = EvidenceClaimSet(claims=(), challenges=())
    context = build_completion_alignment_revision_context(
        resolved_claim_set=empty,
        bindings=(),
        prior_completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
    )
    assert context is not None
    assert context.mismatch_reason is (
        CompletionAlignmentMismatchReason.SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE
    )
    assert context.supported_hypothesis_id is None


def test_builder_aligned_state_returns_none() -> None:
    context = build_completion_alignment_revision_context(
        resolved_claim_set=_authoritative_h3_resolved_claim_set(),
        bindings=_H3_BINDING,
        prior_completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
    )
    assert context is None


def test_builder_uses_critic_resolved_claims_not_pending_model_state() -> None:
    resolved = _critic_resolved_h3_claim_set()
    assert any(claim.resolution is ClaimResolution.SUPPORTED for claim in resolved.claims)
    context = build_completion_alignment_revision_context(
        resolved_claim_set=resolved,
        bindings=_H3_BINDING,
        prior_completion_mode=COMPLETION_UNRESOLVED,
    )
    assert context is not None
    assert context.supported_hypothesis_id is HypothesisId.H3


def test_multiple_supported_hypotheses_fail_closed() -> None:
    claim_set = EvidenceClaimSet(
        claims=(
            EvidenceBackedClaim(
                claim_id=H3_CLAIM_ID,
                statement="H3 supported.",
                claim_kind=DIAGNOSIS_KIND,
                supporting_evidence_ids=(TELEMETRY_EVIDENCE_ID,),
                resolution=ClaimResolution.SUPPORTED,
            ),
            EvidenceBackedClaim(
                claim_id=INITIAL_CLAIM_ID,
                statement="H1 supported.",
                claim_kind=DIAGNOSIS_KIND,
                supporting_evidence_ids=(TELEMETRY_EVIDENCE_ID,),
                resolution=ClaimResolution.SUPPORTED,
            ),
        ),
        challenges=(),
    )
    bindings = (
        ClaimHypothesisBinding(claim_id=str(H3_CLAIM_ID), hypothesis_id="H3"),
        ClaimHypothesisBinding(claim_id=str(INITIAL_CLAIM_ID), hypothesis_id="H1"),
    )
    with pytest.raises(ValueError, match="multiple authoritative supported diagnosis"):
        resolve_authoritative_supported_hypothesis(claim_set, bindings)


def test_renderer_includes_authoritative_semantics_without_prescribing_answer() -> None:
    context = CompletionAlignmentRevisionContext(
        mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        supported_hypothesis_id=HypothesisId.H3,
        supported_resolution=ClaimResolution.SUPPORTED,
    )
    rendered = "\n".join(render_completion_alignment_revision_context(context))
    lowered = rendered.lower()
    assert "authoritative" in lowered
    assert "h3" in lowered
    assert "supported" in lowered
    assert "alignment mismatch" in lowered
    assert "set completion" not in lowered
    assert "choose supported_diagnosis" not in lowered


def test_renderer_is_typed_source_not_feedback_parser() -> None:
    context = CompletionAlignmentRevisionContext(
        mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        supported_hypothesis_id=HypothesisId.H3,
        supported_resolution=ClaimResolution.SUPPORTED,
    )
    rendered = render_completion_alignment_revision_context(context)
    assert UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR not in rendered


def test_s0_s1_distinction_in_revision_messages() -> None:
    prior_proposal = IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H3",
                disposition=HypothesisDisposition.INSUFFICIENT_EVIDENCE,
                summary="H3 remains insufficient without decisive telemetry.",
            ),
        ),
        preferred_hypothesis_id="H3",
        uncertainty_class="decisive_gap",
        information_gaps=("telemetry unavailable",),
        claim_proposals=(
            ClaimProposal(
                hypothesis_id="H3",
                statement="H3 not yet supported.",
                claim_kind=str(DIAGNOSIS_KIND),
            ),
        ),
        completion_intent=CompletionIntent.UNRESOLVED,
        action_objective="remain unresolved",
        unresolved_reason="Telemetry unavailable for incident window.",
    )
    authoritative = CompletionAlignmentRevisionContext(
        mismatch_reason=CompletionAlignmentMismatchReason.UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS,
        supported_hypothesis_id=HypothesisId.H3,
        supported_resolution=ClaimResolution.SUPPORTED,
    )
    messages = build_reasoning_messages(
        evidence_nodes=(),
        prior_state=PriorInvestigationState(
            evidence_nodes=(),
            reasoning_proposal=prior_proposal,
            claim_set=None,
            claim_hypothesis_bindings=(),
            completion_intent=CompletionIntent.UNRESOLVED,
            summary="",
        ),
        critic_feedback=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
        is_revision=True,
        evidence_phase_context=EvidencePhaseContext(
            stop_reason="planner_final_answer",
            additional_evidence_gathering_allowed=False,
        ),
        alignment_revision_context=authoritative,
    )
    system_content = messages[0].content or ""
    prior_index = system_content.index("Prior model reasoning")
    authoritative_index = system_content.index("Authoritative validation state")
    critic_index = system_content.index("Critic feedback requiring incremental correction")
    revision_index = system_content.index("Revision contract:")
    assert prior_index < authoritative_index < critic_index < revision_index
    assert "insufficient" in system_content.lower()
    assert "supported hypothesis: H3" in system_content


def test_no_revision_context_on_aligned_first_pass_messages() -> None:
    messages = build_reasoning_messages(
        evidence_nodes=(),
        prior_state=PriorInvestigationState(
            evidence_nodes=(),
            reasoning_proposal=None,
            claim_set=None,
            claim_hypothesis_bindings=(),
            completion_intent=None,
            summary="",
        ),
        critic_feedback=None,
        is_revision=False,
        evidence_phase_context=EvidencePhaseContext(
            stop_reason="planner_final_answer",
            additional_evidence_gathering_allowed=False,
        ),
    )
    system_content = messages[0].content or ""
    assert "Authoritative validation state" not in system_content
