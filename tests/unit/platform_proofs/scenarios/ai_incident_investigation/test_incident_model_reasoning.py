# © Artur Czarnecki. All rights reserved.

"""APP-2BC-R1 model-owned reasoning, critic authority, and claim semantics tests."""

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_runtime_bundle

import inspect

import pytest

from intergrax.contracts.evidence_claims import ClaimResolution, EvidenceClaimSet
from intergrax.contracts.validation import ValidationResult
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    ClaimHypothesisBinding,
    ClaimProposal,
    CompletionIntent,
    HypothesisDisposition,
    HypothesisProposal,
    IncidentReasoningProposal,
    PriorInvestigationState,
    ReasoningProposalValidationError,
    build_evidence_reference_contract,
    build_investigation_summary,
    build_reasoning_messages,
    completion_mode_from_proposal,
    convert_proposal_to_pending_claims,
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
    INITIAL_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    H3_FORGED_WITHOUT_TELEMETRY_ERROR,
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


def _h3_supported_proposal(*, supporting_evidence_ids: tuple[str, ...]) -> IncidentReasoningProposal:
    workload = str(WORKLOAD_EVIDENCE_ID)
    comparison = str(COMPARISON_EVIDENCE_ID)
    telemetry = str(TELEMETRY_EVIDENCE_ID)
    return IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H3",
                disposition=HypothesisDisposition.SUPPORTED,
                summary="Equipment degradation supported.",
                supporting_evidence_ids=(workload, comparison, telemetry),
            ),
        ),
        preferred_hypothesis_id="H3",
        uncertainty_class="bounded",
        claim_proposals=(
            ClaimProposal(
                hypothesis_id="H3",
                statement="H3 diagnosis without explicit telemetry citation.",
                claim_kind=str(DIAGNOSIS_KIND),
                supporting_evidence_ids=supporting_evidence_ids,
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
    workload = str(WORKLOAD_EVIDENCE_ID)
    proposals = {
        "H1": ClaimProposal(
            hypothesis_id="H1",
            statement="Overload hypothesis H1 pending distinguishing evidence.",
            claim_kind=str(DIAGNOSIS_KIND),
            supporting_evidence_ids=(workload,),
        ),
        "H2": ClaimProposal(
            hypothesis_id="H2",
            statement="Statement mentions H2 but binding is explicit.",
            claim_kind=str(DIAGNOSIS_KIND),
            supporting_evidence_ids=(workload,),
        ),
        "H3": ClaimProposal(
            hypothesis_id="H3",
            statement="Equipment degradation hypothesis H3 pending telemetry.",
            claim_kind=str(DIAGNOSIS_KIND),
            supporting_evidence_ids=(workload,),
        ),
    }
    return IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H1",
                disposition=HypothesisDisposition.PLAUSIBLE,
                summary="Workload-throughput correlation observed.",
                supporting_evidence_ids=(workload,),
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


def test_model_proposal_converts_to_pending_claims() -> None:
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(),
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert conversion.claim_set.claims[0].resolution is ClaimResolution.PENDING
    assert conversion.bindings[0].hypothesis_id == "H1"
    assert str(conversion.claim_set.claims[0].claim_id) != str(INITIAL_CLAIM_ID)


def test_reordered_claims_preserve_hypothesis_binding() -> None:
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H3", "H1", "H2")),
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert [binding.hypothesis_id for binding in conversion.bindings] == ["H3", "H1", "H2"]


def test_missing_h1_claim_allowed() -> None:
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H2", "H3")),
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert {binding.hypothesis_id for binding in conversion.bindings} == {"H2", "H3"}


def test_claim_ids_are_application_minted() -> None:
    first = convert_proposal_to_pending_claims(
        _sample_proposal(),
        prior_claim_set=None,
        critic_feedback=None,
    )
    second = convert_proposal_to_pending_claims(
        _sample_proposal(),
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert first.claim_set.claims[0].claim_id != second.claim_set.claims[0].claim_id


def test_model_cannot_control_resolution_via_proposal_conversion() -> None:
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(),
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert all(claim.resolution is ClaimResolution.PENDING for claim in conversion.claim_set.claims)


def test_unknown_evidence_ref_rejected() -> None:
    proposal = _sample_proposal()
    mutated = proposal.model_copy(
        update={
            "claim_proposals": (
                ClaimProposal(
                    hypothesis_id="H1",
                    statement="bad refs",
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=("evidence.unknown.node",),
                ),
            )
        }
    )
    with pytest.raises(ReasoningProposalValidationError, match="unknown evidence"):
        validate_reasoning_proposal(mutated, evidence_nodes=())


def test_hypothesis_alias_evidence_ref_rejected() -> None:
    """Live-shaped llama output using E1 aliases must stay rejected."""
    workload = str(WORKLOAD_EVIDENCE_ID)
    evidence_nodes = ({"evidence_id": workload, "payload": {}},)
    proposal = IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H1",
                disposition=HypothesisDisposition.PLAUSIBLE,
                summary="Overload hypothesis.",
                supporting_evidence_ids=("E1",),
            ),
        ),
        preferred_hypothesis_id="H1",
        uncertainty_class="high",
        information_gaps=("comparison evidence",),
        claim_proposals=(
            ClaimProposal(
                hypothesis_id="H1",
                statement="Overload pending distinguishing evidence.",
                claim_kind=str(DIAGNOSIS_KIND),
                supporting_evidence_ids=(workload,),
            ),
        ),
        completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
        action_objective="gather distinguishing evidence",
    )
    with pytest.raises(
        ReasoningProposalValidationError,
        match="unknown evidence reference in hypothesis: E1",
    ):
        validate_reasoning_proposal(proposal, evidence_nodes=evidence_nodes)


def test_hypothesis_evid_dash_alias_rejected() -> None:
    """Live-shaped llama3.1 output using EVID-NNN aliases must stay rejected."""
    workload = str(WORKLOAD_EVIDENCE_ID)
    evidence_nodes = ({"evidence_id": workload, "payload": {}},)
    proposal = IncidentReasoningProposal(
        hypotheses=(
            HypothesisProposal(
                hypothesis_id="H1",
                disposition=HypothesisDisposition.PLAUSIBLE,
                summary="Overload hypothesis.",
                supporting_evidence_ids=("EVID-001",),
            ),
        ),
        preferred_hypothesis_id="H1",
        uncertainty_class="high",
        information_gaps=("comparison evidence",),
        claim_proposals=(
            ClaimProposal(
                hypothesis_id="H1",
                statement="Overload pending distinguishing evidence.",
                claim_kind=str(DIAGNOSIS_KIND),
                supporting_evidence_ids=(workload,),
            ),
        ),
        completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
        action_objective="gather distinguishing evidence",
    )
    with pytest.raises(
        ReasoningProposalValidationError,
        match="unknown evidence reference in hypothesis: EVID-001",
    ):
        validate_reasoning_proposal(proposal, evidence_nodes=evidence_nodes)


def test_reasoning_prompt_includes_evidence_reference_contract() -> None:
    workload = str(WORKLOAD_EVIDENCE_ID)
    telemetry = str(TELEMETRY_EVIDENCE_ID)
    evidence_nodes = (
        {"evidence_id": workload, "payload": {}},
        {"evidence_id": telemetry, "payload": {}},
    )
    contract = build_evidence_reference_contract(evidence_nodes)
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
    assert contract in system_prompt
    assert contract in user_prompt
    assert workload in system_prompt
    assert telemetry in system_prompt
    assert "Allowed evidence IDs:" in system_prompt
    assert str(WORKLOAD_EVIDENCE_ID) not in system_prompt or workload in system_prompt


def test_evidence_reference_contract_excludes_invented_example_id() -> None:
    actual_a = "evidence.actual.id.a"
    actual_b = "evidence.actual.id.b"
    contract = build_evidence_reference_contract(
        (
            {"evidence_id": actual_a, "payload": {}},
            {"evidence_id": actual_b, "payload": {}},
        )
    )
    assert actual_a in contract
    assert actual_b in contract
    assert "for example" not in contract.lower()
    assert "e.g." not in contract.lower()
    assert str(WORKLOAD_EVIDENCE_ID) not in contract


def test_evidence_reference_contract_empty_whitelist() -> None:
    contract = build_evidence_reference_contract(())
    assert "Allowed evidence IDs: none." in contract
    assert "Do not invent an evidence ID." in contract


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
    assert "Allowed evidence IDs: none." in system_prompt
    assert "Do not invent an evidence ID." in system_prompt


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


def test_model_omitted_telemetry_not_silently_inserted() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    comparison = str(COMPARISON_EVIDENCE_ID)
    telemetry = str(TELEMETRY_EVIDENCE_ID)
    workload = str(WORKLOAD_EVIDENCE_ID)
    proposal = _h3_supported_proposal(supporting_evidence_ids=(workload, comparison))
    validate_reasoning_proposal(proposal, evidence_nodes=evidence_nodes)

    h3_claim = proposal.claim_proposals[0]
    assert telemetry not in h3_claim.supporting_evidence_ids

    conversion = convert_proposal_to_pending_claims(
        proposal,
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert telemetry not in conversion.claim_set.claims[0].supporting_evidence_ids

    _, validation = _validate_h3_claim_set(proposal, evidence_nodes=evidence_nodes)
    assert not validation.valid
    assert H3_FORGED_WITHOUT_TELEMETRY_ERROR in validation.errors


def test_evidence_availability_does_not_imply_claim_support() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    assert any(
        str(node.get("evidence_id")) == str(TELEMETRY_EVIDENCE_ID) for node in evidence_nodes
    )
    proposal = _h3_supported_proposal(
        supporting_evidence_ids=(str(WORKLOAD_EVIDENCE_ID), str(COMPARISON_EVIDENCE_ID)),
    )
    _, validation = _validate_h3_claim_set(proposal, evidence_nodes=evidence_nodes)
    assert not validation.valid
    assert H3_FORGED_WITHOUT_TELEMETRY_ERROR in validation.errors


def test_revision_with_explicit_citations_accepted() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    invalid = _h3_supported_proposal(
        supporting_evidence_ids=(str(WORKLOAD_EVIDENCE_ID), str(COMPARISON_EVIDENCE_ID)),
    )
    _, invalid_validation = _validate_h3_claim_set(invalid, evidence_nodes=evidence_nodes)
    assert not invalid_validation.valid

    revised = _h3_supported_proposal(
        supporting_evidence_ids=(
            str(WORKLOAD_EVIDENCE_ID),
            str(COMPARISON_EVIDENCE_ID),
            str(TELEMETRY_EVIDENCE_ID),
        ),
    ).model_copy(
        update={
            "claim_proposals": (
                ClaimProposal(
                    hypothesis_id="H3",
                    statement="Revised H3 diagnosis cites comparison and telemetry explicitly.",
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=(
                        str(WORKLOAD_EVIDENCE_ID),
                        str(COMPARISON_EVIDENCE_ID),
                        str(TELEMETRY_EVIDENCE_ID),
                    ),
                    replaces_prior_claim=True,
                ),
            )
        }
    )
    _, revised_validation = _validate_h3_claim_set(revised, evidence_nodes=evidence_nodes)
    assert revised_validation.valid


def test_revision_still_omitting_required_citation_remains_invalid() -> None:
    evidence_nodes = _resolved_evidence_nodes()
    still_invalid = _h3_supported_proposal(
        supporting_evidence_ids=(str(WORKLOAD_EVIDENCE_ID), str(COMPARISON_EVIDENCE_ID)),
    ).model_copy(
        update={
            "claim_proposals": (
                ClaimProposal(
                    hypothesis_id="H3",
                    statement="Revision still omits telemetry citation.",
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=(
                        str(WORKLOAD_EVIDENCE_ID),
                        str(COMPARISON_EVIDENCE_ID),
                    ),
                    replaces_prior_claim=True,
                ),
            )
        }
    )
    _, validation = _validate_h3_claim_set(still_invalid, evidence_nodes=evidence_nodes)
    assert not validation.valid
    assert H3_FORGED_WITHOUT_TELEMETRY_ERROR in validation.errors


def test_bounded_summary_requires_explicit_model_citations() -> None:
    proposal = _h3_supported_proposal(
        supporting_evidence_ids=(str(WORKLOAD_EVIDENCE_ID), str(COMPARISON_EVIDENCE_ID)),
    )
    summary = build_investigation_summary(proposal, is_revision=False)
    assert "best-supported initiating cause" not in summary


def test_revision_prompt_requires_explicit_citation_repair_by_model() -> None:
    initial_nodes = ({"evidence_id": str(WORKLOAD_EVIDENCE_ID), "payload": {}},)
    revision_nodes = (
        {"evidence_id": str(WORKLOAD_EVIDENCE_ID), "payload": {}},
        {"evidence_id": str(TELEMETRY_EVIDENCE_ID), "payload": {}},
    )
    prior_proposal = _sample_proposal()
    messages = build_reasoning_messages(
        evidence_nodes=revision_nodes,
        prior_state=PriorInvestigationState(
            evidence_nodes=initial_nodes,
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
    assert str(TELEMETRY_EVIDENCE_ID) in prompt
    assert "Revision contract:" in prompt
    assert "Do not assume the platform will add or repair claim citations." in prompt
    assert "Do not cite evidence that is not in the current allowed list." in prompt
    assert str(COMPARISON_EVIDENCE_ID) not in prompt


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
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(),
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
                "evidence_nodes": [],
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
    initial = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H1", "H2", "H3")),
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
                    supporting_evidence_ids=(str(WORKLOAD_EVIDENCE_ID),),
                    replaces_prior_claim=True,
                ),
            )
        }
    )
    revised = convert_proposal_to_pending_claims(
        proposal,
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
    conversion = convert_proposal_to_pending_claims(
        _sample_proposal(claim_order=("H1", "H2", "H3")),
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
