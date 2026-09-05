# © Artur Czarnecki. All rights reserved.

"""APP-2BC-R1 model-owned reasoning, critic authority, and claim semantics tests."""

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_runtime_bundle

import inspect

import pytest

from intergrax.contracts.evidence_claims import ClaimResolution, EvidenceClaimSet
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    ClaimHypothesisBinding,
    ClaimProposal,
    CompletionIntent,
    EVIDENCE_REFERENCE_CONTRACT,
    HypothesisDisposition,
    HypothesisProposal,
    IncidentReasoningProposal,
    PriorInvestigationState,
    ReasoningProposalValidationError,
    build_reasoning_messages,
    convert_proposal_to_pending_claims,
    validate_reasoning_proposal,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    DIAGNOSIS_KIND,
    H2_CLAIM_ID,
    H3_CLAIM_ID,
    INITIAL_CLAIM_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    apply_critic_claim_resolutions,
    validate_claim_set_against_observations,
)

pytestmark = pytest.mark.unit


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
    messages = build_reasoning_messages(
        evidence_nodes=({"evidence_id": workload, "payload": {}},),
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
    assert EVIDENCE_REFERENCE_CONTRACT in system_prompt
    assert EVIDENCE_REFERENCE_CONTRACT in user_prompt
    assert workload in system_prompt


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
    assert all(
        claim.get("resolution") != ClaimResolution.PENDING.value
        for claim in result.claim_set.get("claims", [])
        if claim.get("claim_kind") == str(DIAGNOSIS_KIND)
    )
