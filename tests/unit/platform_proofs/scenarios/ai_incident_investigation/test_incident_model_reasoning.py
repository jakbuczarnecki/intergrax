# © Artur Czarnecki. All rights reserved.

"""APP-2BC-R1 model-owned reasoning, critic authority, and claim semantics tests."""

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_runtime_bundle

import copy
import inspect

import pytest

from intergrax.contracts.evidence_claims import ClaimResolution, EvidenceClaimSet
from intergrax.contracts.validation import ValidationResult
from platform_proofs.scenarios.ai_incident_investigation.application.claim_evidence_attribution import (
    attribute_claim_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import (
    derive_hypothesis_dispositions,
    observations_from_evidence_nodes,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    ClaimHypothesisBinding,
    ClaimProposal,
    CompletionIntent,
    HypothesisDisposition,
    HypothesisProposal,
    IncidentReasoningProposal,
    PriorInvestigationState,
    ReasoningProposalValidationError,
    build_investigation_summary,
    build_reasoning_messages,
    completion_mode_from_proposal,
    convert_proposal_to_pending_claims,
    latest_active_claim_for_hypothesis,
    validate_reasoning_proposal,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import build_resolved_fixture
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    COMPLETION_NEED_MORE_EVIDENCE,
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
    DIAGNOSIS_KIND,
    H2_CLAIM_ID,
    H3_CLAIM_ID,
    INCIDENT_EVIDENCE_IDS,
    INITIAL_CLAIM_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    apply_critic_claim_resolutions,
    validate_claim_set_against_observations,
)

pytestmark = pytest.mark.unit


def _resolved_evidence_nodes() -> tuple[dict[str, object], ...]:
    fixture = build_resolved_fixture()
    return (
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
                "record_valid_from": fixture.staffing_preliminary.record_valid_for.observed_from.isoformat(),
                "record_valid_to": fixture.staffing_preliminary.record_valid_for.observed_to.isoformat(),
                "window_observed_from": fixture.staffing_preliminary.window.observed_from.isoformat(),
                "window_observed_to": fixture.staffing_preliminary.window.observed_to.isoformat(),
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
                "complex_assembly_throughput_pct": fixture.telemetry.complex_assembly_throughput_pct,
                "baseline_throughput_pct": fixture.telemetry.baseline_throughput_pct,
                "admissible": True,
            },
        },
    )


def _h3_semantic_proposal() -> IncidentReasoningProposal:
    return IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H3",
                disposition=HypothesisDisposition.SUPPORTED,
                summary="Equipment degradation supported.",
            ),
        ),
        preferred_hypothesis_id="H3",
        uncertainty_class="bounded",
        claim_proposals=(
            ClaimProposal(
                hypothesis_id="H3",
                statement="H3 diagnosis without model-owned evidence IDs.",
                claim_kind=str(DIAGNOSIS_KIND),
            ),
        ),
        completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
        action_objective="propose bounded H3 diagnosis",
    )


def _validate_h3_claim_set(
    proposal: IncidentReasoningProposal,
    *,
    evidence_nodes: tuple[dict[str, object], ...],
) -> tuple[EvidenceClaimSet, ValidationResult]:
    conversion = convert_proposal_to_pending_claims(
        proposal,
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    resolved = apply_critic_claim_resolutions(
        conversion.claim_set,
        {
            "evidence_nodes": list(evidence_nodes),
            "claim_hypothesis_bindings": [
                binding.model_dump(mode="json") for binding in conversion.bindings
            ],
        },
        bindings=conversion.bindings,
    )
    validation = validate_claim_set_against_observations(
        resolved,
        {
            "claim_set": resolved.model_dump(mode="json"),
            "claim_hypothesis_bindings": [
                binding.model_dump(mode="json") for binding in conversion.bindings
            ],
            "evidence_nodes": list(evidence_nodes),
            "active_hypothesis": "H3",
            "completion_mode": COMPLETION_SUPPORTED_DIAGNOSIS,
        },
        bindings=conversion.bindings,
    )
    return resolved, validation


def _sample_proposal(*, claim_order: tuple[str, ...] = ("H1",)) -> IncidentReasoningProposal:
    proposals = {
        "H1": ClaimProposal(
            hypothesis_id="H1",
            statement="Overload hypothesis H1 pending distinguishing evidence.",
            claim_kind=str(DIAGNOSIS_KIND),
        ),
        "H2": ClaimProposal(
            hypothesis_id="H2",
            statement="Statement mentions H2 but binding is explicit.",
            claim_kind=str(DIAGNOSIS_KIND),
        ),
        "H3": ClaimProposal(
            hypothesis_id="H3",
            statement="Equipment degradation hypothesis H3 pending telemetry.",
            claim_kind=str(DIAGNOSIS_KIND),
        ),
    }
    return IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H1",
                disposition=HypothesisDisposition.PLAUSIBLE,
                summary="Workload-throughput correlation observed.",
            ),
        ),
        preferred_hypothesis_id="H1",
        uncertainty_class="high",
        information_gaps=("comparison evidence",),
        claim_proposals=tuple(proposals[item] for item in claim_order),
        completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
        action_objective="gather distinguishing evidence",
    )


def test_claim_proposal_requires_hypothesis_id() -> None:
    with pytest.raises(Exception):
        ClaimProposal.model_validate(
            {
                "statement": "missing hypothesis",
                "claim_kind": str(DIAGNOSIS_KIND),
            }
        )


def test_claim_proposal_has_no_evidence_id_fields() -> None:
    fields = set(ClaimProposal.model_fields)
    assert "supporting_evidence_ids" not in fields
    assert "contradicting_evidence_ids" not in fields


def test_hypothesis_proposal_has_no_evidence_id_fields() -> None:
    fields = set(HypothesisProposal.model_fields)
    assert "supporting_evidence_ids" not in fields
    assert "contradicting_evidence_ids" not in fields


def test_model_proposal_converts_to_pending_claims() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert conversion.claim_set.claims[0].resolution is ClaimResolution.PENDING
    assert conversion.bindings[0].hypothesis_id == "H1"
    assert str(conversion.claim_set.claims[0].claim_id) != str(INITIAL_CLAIM_ID)


def test_reordered_claims_preserve_hypothesis_binding() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H3", "H1", "H2")),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert [binding.hypothesis_id for binding in conversion.bindings] == ["H3", "H1", "H2"]


def test_missing_h1_claim_allowed() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H2", "H3")),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert {binding.hypothesis_id for binding in conversion.bindings} == {"H2", "H3"}


def test_claim_ids_are_application_minted() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    first = convert_proposal_to_pending_claims(
        _sample_proposal(),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    second = convert_proposal_to_pending_claims(
        _sample_proposal(),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert first.claim_set.claims[0].claim_id != second.claim_set.claims[0].claim_id


def test_model_cannot_control_resolution_via_proposal_conversion() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert all(claim.resolution is ClaimResolution.PENDING for claim in conversion.claim_set.claims)


def test_reasoning_prompt_excludes_evidence_id_copy_contract() -> None:
    workload = str(WORKLOAD_EVIDENCE_ID)
    telemetry = str(TELEMETRY_EVIDENCE_ID)
    evidence_nodes = (
        {"evidence_id": workload, "payload": {}},
        {"evidence_id": telemetry, "payload": {}},
    )
    messages = build_reasoning_messages(
        evidence_nodes=evidence_nodes,
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
    )
    system_prompt = messages[0].content or ""
    user_prompt = messages[1].content or ""
    assert "Do not emit evidence_id fields" in system_prompt
    assert workload in system_prompt
    assert telemetry in system_prompt
    assert "supporting_evidence_ids" not in system_prompt
    assert "supporting_evidence_ids" not in user_prompt


def test_reasoning_prompt_empty_evidence_instructs_no_invention() -> None:
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
    )
    system_prompt = messages[0].content or ""
    assert "Gathered evidence IDs: none yet." in system_prompt


def test_workload_example_id_absent_from_prompt_without_evidence() -> None:
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
    )
    prompt = (messages[0].content or "") + (messages[1].content or "")
    assert str(WORKLOAD_EVIDENCE_ID) not in prompt


def test_semantic_output_repair_removed() -> None:
    import platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning as mod

    assert not hasattr(mod, "normalize_supported_diagnosis_claim_evidence")


def test_zero_model_ids_h3_gets_deterministic_attribution_and_passes() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    proposal = _h3_semantic_proposal()
    original = copy.deepcopy(proposal)
    validate_reasoning_proposal(proposal, evidence_nodes=evidence_nodes)

    conversion = convert_proposal_to_pending_claims(
        proposal,
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    h3_claim = conversion.claim_set.claims[0]
    assert str(TELEMETRY_EVIDENCE_ID) in h3_claim.supporting_evidence_ids
    assert str(COMPARISON_EVIDENCE_ID) in h3_claim.supporting_evidence_ids

    resolved, validation = _validate_h3_claim_set(proposal, evidence_nodes=evidence_nodes)
    assert proposal == original
    assert validation.valid
    h3_resolved = resolved.claims[0]
    assert h3_resolved.resolution is ClaimResolution.SUPPORTED


def test_telemetry_exists_without_degradation_does_not_support_h3() -> None:
    fixture = build_resolved_fixture()
    evidence_nodes = (
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
                "signal_state": "healthy",
                "complex_assembly_throughput_pct": 90.0,
                "baseline_throughput_pct": 91.0,
                "admissible": True,
            },
        },
    )
    proposal = _h3_semantic_proposal()
    conversion = convert_proposal_to_pending_claims(
        proposal,
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    h3_claim = conversion.claim_set.claims[0]
    assert str(TELEMETRY_EVIDENCE_ID) not in h3_claim.supporting_evidence_ids

    resolved, validation = _validate_h3_claim_set(proposal, evidence_nodes=evidence_nodes)
    assert not validation.valid
    h3_resolved = resolved.claims[0]
    assert h3_resolved.resolution is not ClaimResolution.SUPPORTED


def test_h1_contradiction_when_comparison_weakens_overload() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    observations = observations_from_evidence_nodes(evidence_nodes, INCIDENT_EVIDENCE_IDS)
    observable_ids = frozenset(str(node["evidence_id"]) for node in evidence_nodes)
    attribution = attribute_claim_evidence("H1", observations, INCIDENT_EVIDENCE_IDS, observable_ids)
    assert str(WORKLOAD_EVIDENCE_ID) in attribution.supporting_evidence_ids
    assert str(THROUGHPUT_EVIDENCE_ID) in attribution.supporting_evidence_ids
    assert str(COMPARISON_EVIDENCE_ID) in attribution.contradicting_evidence_ids


def test_h2_staffing_shortage_support_attribution() -> None:
    fixture = build_resolved_fixture()
    evidence_nodes = (
        {
            "evidence_id": str(STAFFING_PRELIMINARY_EVIDENCE_ID),
            "payload": {
                "scheduled_headcount": 4,
                "required_headcount": 6,
                "record_valid_from": fixture.staffing_preliminary.record_valid_for.observed_from.isoformat(),
                "record_valid_to": fixture.staffing_preliminary.record_valid_for.observed_to.isoformat(),
                "window_observed_from": fixture.staffing_preliminary.window.observed_from.isoformat(),
                "window_observed_to": fixture.staffing_preliminary.window.observed_to.isoformat(),
            },
        },
        {
            "evidence_id": str(STAFFING_ATTENDANCE_EVIDENCE_ID),
            "payload": {"confirmed_headcount": 4},
        },
    )
    observations = observations_from_evidence_nodes(evidence_nodes, INCIDENT_EVIDENCE_IDS)
    observable_ids = frozenset(str(node["evidence_id"]) for node in evidence_nodes)
    attribution = attribute_claim_evidence("H2", observations, INCIDENT_EVIDENCE_IDS, observable_ids)
    assert str(STAFFING_ATTENDANCE_EVIDENCE_ID) in attribution.supporting_evidence_ids


def test_h2_attendance_meets_required_contradiction_attribution() -> None:
    fixture = build_resolved_fixture()
    evidence_nodes = (
        {
            "evidence_id": str(STAFFING_PRELIMINARY_EVIDENCE_ID),
            "payload": {
                "scheduled_headcount": 4,
                "required_headcount": 6,
                "record_valid_from": fixture.staffing_preliminary.record_valid_for.observed_from.isoformat(),
                "record_valid_to": fixture.staffing_preliminary.record_valid_for.observed_to.isoformat(),
                "window_observed_from": fixture.staffing_preliminary.window.observed_from.isoformat(),
                "window_observed_to": fixture.staffing_preliminary.window.observed_to.isoformat(),
            },
        },
        {
            "evidence_id": str(STAFFING_ATTENDANCE_EVIDENCE_ID),
            "payload": {"confirmed_headcount": 6},
        },
    )
    observations = observations_from_evidence_nodes(evidence_nodes, INCIDENT_EVIDENCE_IDS)
    observable_ids = frozenset(str(node["evidence_id"]) for node in evidence_nodes)
    attribution = attribute_claim_evidence("H2", observations, INCIDENT_EVIDENCE_IDS, observable_ids)
    assert str(STAFFING_PRELIMINARY_EVIDENCE_ID) in attribution.supporting_evidence_ids
    assert str(STAFFING_ATTENDANCE_EVIDENCE_ID) in attribution.contradicting_evidence_ids


def test_attribution_never_emits_unobservable_ids() -> None:
    evidence_nodes = _resolved_evidence_nodes()[:2]
    observations = observations_from_evidence_nodes(evidence_nodes, INCIDENT_EVIDENCE_IDS)
    observable_ids = frozenset(str(node["evidence_id"]) for node in evidence_nodes)
    attribution = attribute_claim_evidence("H3", observations, INCIDENT_EVIDENCE_IDS, observable_ids)
    for evidence_id in (
        *attribution.supporting_evidence_ids,
        *attribution.contradicting_evidence_ids,
    ):
        assert str(evidence_id) in observable_ids


def test_model_proposal_immutable_after_conversion() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    proposal = _h3_semantic_proposal()
    original = copy.deepcopy(proposal)
    convert_proposal_to_pending_claims(
        proposal,
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert proposal == original


def test_provider_independent_semantic_proposals_share_attribution() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    proposal_a = _h3_semantic_proposal()
    proposal_b = proposal_a.model_copy(
        update={
            "claim_proposals": (
                ClaimProposal(
                    hypothesis_id="H3",
                    statement="Alternate wording — same semantics, zero evidence IDs.",
                    claim_kind=str(DIAGNOSIS_KIND),
                    rationale="provider-neutral fixture",
                ),
            )
        }
    )
    conversion_a = convert_proposal_to_pending_claims(
        proposal_a,
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    conversion_b = convert_proposal_to_pending_claims(
        proposal_b,
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    claim_a = conversion_a.claim_set.claims[0]
    claim_b = conversion_b.claim_set.claims[0]
    assert claim_a.supporting_evidence_ids == claim_b.supporting_evidence_ids
    assert claim_a.contradicting_evidence_ids == claim_b.contradicting_evidence_ids


def test_bounded_summary_uses_semantic_h3_completion_intent() -> None:
    proposal = _h3_semantic_proposal()
    summary = build_investigation_summary(proposal, is_revision=False)
    assert "best-supported initiating cause" in summary


def test_revision_prompt_does_not_require_model_evidence_citations() -> None:
    revision_nodes = (
        {"evidence_id": str(WORKLOAD_EVIDENCE_ID), "payload": {}},
        {"evidence_id": str(TELEMETRY_EVIDENCE_ID), "payload": {}},
    )
    prior_proposal = _sample_proposal()
    messages = build_reasoning_messages(
        evidence_nodes=revision_nodes,
        prior_state=PriorInvestigationState(
            evidence_nodes=revision_nodes,
            reasoning_proposal=prior_proposal,
            claim_set=None,
            claim_hypothesis_bindings=(),
            completion_intent=CompletionIntent.NEED_MORE_EVIDENCE,
            summary="prior summary",
        ),
        critic_feedback=("unsupported inference",),
        is_revision=True,
    )
    prompt = messages[0].content or ""
    assert "Revision contract:" in prompt
    assert "explicitly cite the relevant gathered evidence IDs" not in prompt
    assert "Do not assume the platform will add or repair claim citations." not in prompt


def test_completion_mode_maps_need_more_evidence_separately() -> None:
    proposal = _sample_proposal().model_copy(
        update={"completion_intent": CompletionIntent.NEED_MORE_EVIDENCE}
    )
    assert completion_mode_from_proposal(proposal) == COMPLETION_NEED_MORE_EVIDENCE


def test_completion_mode_maps_unresolved_and_supported_diagnosis() -> None:
    unresolved = _sample_proposal().model_copy(
        update={
            "completion_intent": CompletionIntent.UNRESOLVED,
            "unresolved_reason": "telemetry unavailable",
            "information_gaps": ("decisive telemetry",),
        }
    )
    supported = _sample_proposal()
    assert completion_mode_from_proposal(unresolved) == COMPLETION_UNRESOLVED
    assert completion_mode_from_proposal(supported) == COMPLETION_SUPPORTED_DIAGNOSIS


def test_reasoning_prompt_requires_unresolved_fields_and_claim_proposals() -> None:
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
    )
    prompt = (messages[0].content or "") + (messages[1].content or "")
    assert "unresolved_reason" in prompt
    assert "information_gaps" in prompt
    assert "claim_proposals must always be non-empty" in prompt
    assert str(DIAGNOSIS_KIND) in prompt


def test_critic_apply_resolutions_rejects_model_self_approval() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    claim_set = EvidenceClaimSet(
        claims=(
            conversion.claim_set.claims[0].model_copy(
                update={"resolution": ClaimResolution.SUPPORTED}
            ),
        ),
        challenges=(),
    )
    with pytest.raises(ValueError, match="model_self_approved"):
        apply_critic_claim_resolutions(
            claim_set,
            {
                "evidence_nodes": list(evidence_nodes),
                "claim_hypothesis_bindings": [
                    binding.model_dump(mode="json") for binding in conversion.bindings
                ],
            },
            bindings=conversion.bindings,
        )


def test_validation_does_not_call_derive_hypothesis_dispositions() -> None:
    source = inspect.getsource(validate_claim_set_against_observations)
    assert "derive_hypothesis_dispositions" not in source


def test_investigator_does_not_call_apply_critic_claim_resolutions() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.application import investigator_agent as mod

    source = inspect.getsource(mod.IncidentInvestigatorAgent.run_step)
    assert "apply_critic_claim_resolutions" not in source
    assert "derive_hypothesis_dispositions" not in source


def test_revision_supersedes_same_hypothesis_only() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    initial = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H1", "H2", "H3")),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    proposal = _sample_proposal(claim_order=("H3",)).model_copy(
        update={
            "claim_proposals": (
                ClaimProposal(
                    hypothesis_id="H3",
                    statement="Revised bounded diagnosis.",
                    claim_kind=str(DIAGNOSIS_KIND),
                    replaces_prior_claim=True,
                ),
            )
        }
    )
    revised = convert_proposal_to_pending_claims(
        proposal,
        evidence_nodes=evidence_nodes,
        prior_claim_set=initial.claim_set,
        prior_bindings=initial.bindings,
        critic_feedback=["unsupported inference"],
    )
    h3_prior = next(binding.claim_id for binding in initial.bindings if binding.hypothesis_id == "H3")
    h1_prior = next(binding.claim_id for binding in initial.bindings if binding.hypothesis_id == "H1")
    revised_h3 = next(
        claim
        for claim in revised.claim_set.claims
        if str(claim.claim_id)
        in {binding.claim_id for binding in revised.bindings if binding.hypothesis_id == "H3"}
    )
    assert revised_h3.supersedes_claim_id == h3_prior
    h1_claim = next(claim for claim in revised.claim_set.claims if str(claim.claim_id) == h1_prior)
    assert h1_claim.supersedes_claim_id is None


def test_legacy_claim_ids_not_used_for_semantic_conversion() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H1", "H2", "H3")),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    claim_ids = {str(claim.claim_id) for claim in conversion.claim_set.claims}
    assert str(INITIAL_CLAIM_ID) not in claim_ids
    assert str(H2_CLAIM_ID) not in claim_ids
    assert str(H3_CLAIM_ID) not in claim_ids


@pytest.mark.asyncio
async def test_application_survives_without_proof_evaluator() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
        OUTCOME_RESOLVED,
        execute_resolved_skeleton,
    )

    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_verdict_passed
    assert result.claim_set
    assert result.evidence_nodes
    bound_claim_ids = {
        binding["claim_id"]
        for binding in result.claim_hypothesis_bindings
    }
    assert all(
        claim.get("resolution") != ClaimResolution.PENDING.value
        for claim in result.claim_set.get("claims", [])
        if claim.get("claim_kind") == str(DIAGNOSIS_KIND)
        and claim.get("claim_id") in bound_claim_ids
    )


@pytest.mark.asyncio
async def test_golden_contract_model_semantics_to_validator_resolved() -> None:
    """DS-E2E-12 golden: semantic proposal → attribution → critic → validator → RESOLVED."""
    from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
        OUTCOME_RESOLVED,
        execute_resolved_skeleton,
    )

    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_verdict_passed
    supported = [
        claim
        for claim in result.claim_set.get("claims", [])
        if claim.get("resolution") == ClaimResolution.SUPPORTED.value
    ]
    assert supported
    assert str(TELEMETRY_EVIDENCE_ID) in supported[-1].get("supporting_evidence_ids", [])
    assert str(COMPARISON_EVIDENCE_ID) in supported[-1].get("supporting_evidence_ids", [])


def _h2_evidence_nodes(
    *,
    scheduled_headcount: int,
    required_headcount: int,
    confirmed_headcount: int,
    stale_schedule: bool,
) -> tuple[dict[str, object], ...]:
    fixture = build_resolved_fixture()
    schedule_payload = {
        "scheduled_headcount": scheduled_headcount,
        "required_headcount": required_headcount,
        "record_valid_from": fixture.staffing_preliminary.record_valid_for.observed_from.isoformat(),
        "record_valid_to": fixture.staffing_preliminary.record_valid_for.observed_to.isoformat(),
        "window_observed_from": fixture.staffing_preliminary.window.observed_from.isoformat(),
        "window_observed_to": fixture.staffing_preliminary.window.observed_to.isoformat(),
    }
    if stale_schedule:
        schedule_payload["record_valid_to"] = fixture.staffing_preliminary.window.observed_from.isoformat()
    return (
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
            "payload": schedule_payload,
        },
        {
            "evidence_id": str(STAFFING_ATTENDANCE_EVIDENCE_ID),
            "payload": {"confirmed_headcount": confirmed_headcount},
        },
    )


def test_h2_layers_agree_on_rejected_when_attendance_meets_required() -> None:
    evidence_nodes = _h2_evidence_nodes(
        scheduled_headcount=4,
        required_headcount=6,
        confirmed_headcount=6,
        stale_schedule=False,
    )
    observations = observations_from_evidence_nodes(evidence_nodes, INCIDENT_EVIDENCE_IDS)
    runtime = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H2",)),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    resolved = apply_critic_claim_resolutions(
        conversion.claim_set,
        {
            "evidence_nodes": list(evidence_nodes),
            "claim_hypothesis_bindings": [
                binding.model_dump(mode="json") for binding in conversion.bindings
            ],
        },
        bindings=conversion.bindings,
    )
    h2_claim = latest_active_claim_for_hypothesis(resolved, conversion.bindings, "H2")
    assert h2_claim is not None
    assert h2_claim.resolution is ClaimResolution.REJECTED
    assert runtime.h2.disposition is ClaimResolution.REJECTED


def test_h2_layers_agree_on_supported_when_shortage_confirmed() -> None:
    evidence_nodes = _h2_evidence_nodes(
        scheduled_headcount=4,
        required_headcount=6,
        confirmed_headcount=4,
        stale_schedule=False,
    )
    observations = observations_from_evidence_nodes(evidence_nodes, INCIDENT_EVIDENCE_IDS)
    runtime = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H2",)),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    resolved = apply_critic_claim_resolutions(
        conversion.claim_set,
        {
            "evidence_nodes": list(evidence_nodes),
            "claim_hypothesis_bindings": [
                binding.model_dump(mode="json") for binding in conversion.bindings
            ],
        },
        bindings=conversion.bindings,
    )
    h2_claim = latest_active_claim_for_hypothesis(resolved, conversion.bindings, "H2")
    assert h2_claim is not None
    assert h2_claim.resolution is ClaimResolution.SUPPORTED
    assert runtime.h2.disposition is ClaimResolution.SUPPORTED


def test_latest_active_claim_follows_h2_revision_lineage() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    initial = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H2",)),
        evidence_nodes=evidence_nodes,
        prior_claim_set=None,
        critic_feedback=None,
    )
    revised = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H2",)).model_copy(
            update={
                "claim_proposals": (
                    ClaimProposal(
                        hypothesis_id="H2",
                        statement="Revised staffing assessment.",
                        claim_kind=str(DIAGNOSIS_KIND),
                        replaces_prior_claim=True,
                    ),
                )
            }
        ),
        evidence_nodes=evidence_nodes,
        prior_claim_set=initial.claim_set,
        prior_bindings=initial.bindings,
        critic_feedback=["unsupported inference"],
    )
    stale_binding = ClaimHypothesisBinding(
        claim_id=next(
            binding.claim_id for binding in initial.bindings if binding.hypothesis_id == "H2"
        ),
        hypothesis_id="H2",
    )
    effective = latest_active_claim_for_hypothesis(
        revised.claim_set,
        (stale_binding,),
        "H2",
    )
    expected = latest_active_claim_for_hypothesis(
        revised.claim_set,
        revised.bindings,
        "H2",
    )
    assert effective is not None
    assert expected is not None
    assert str(effective.claim_id) == str(expected.claim_id)
    assert effective.supersedes_claim_id == stale_binding.claim_id
