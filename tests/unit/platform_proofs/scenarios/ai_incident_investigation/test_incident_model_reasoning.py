# © Artur Czarnecki. All rights reserved.

"""APP-2BC model-owned reasoning and critic independence tests."""

from __future__ import annotations

import inspect

import pytest

from intergrax.contracts.evidence_claims import ClaimResolution, EvidenceClaimSet
from platform_proofs.scenarios.ai_incident_investigation.incident_reasoning import (
    ClaimProposal,
    CompletionIntent,
    HypothesisDisposition,
    HypothesisProposal,
    IncidentReasoningProposal,
    ReasoningProposalValidationError,
    convert_proposal_to_pending_claims,
    validate_reasoning_proposal,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario_contract import (
    DIAGNOSIS_KIND,
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.validation import (
    apply_critic_claim_resolutions,
    validate_claim_set_against_observations,
)

pytestmark = pytest.mark.unit


def _sample_proposal() -> IncidentReasoningProposal:
    workload = str(WORKLOAD_EVIDENCE_ID)
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
        claim_proposals=(
            ClaimProposal(
                statement="Overload hypothesis H1 pending distinguishing evidence.",
                claim_kind=str(DIAGNOSIS_KIND),
                supporting_evidence_ids=(workload,),
            ),
        ),
        completion_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
        action_objective="gather distinguishing evidence",
    )


def test_model_proposal_converts_to_pending_claims() -> None:
    proposal = _sample_proposal()
    claim_set = convert_proposal_to_pending_claims(proposal, prior_claim_set=None, critic_feedback=None)
    assert claim_set.claims[0].resolution is ClaimResolution.PENDING
    assert claim_set.claims[0].claim_id == INITIAL_CLAIM_ID


def test_model_cannot_control_resolution_via_proposal_conversion() -> None:
    claim_set = convert_proposal_to_pending_claims(
        _sample_proposal(),
        prior_claim_set=None,
        critic_feedback=None,
    )
    assert all(claim.resolution is ClaimResolution.PENDING for claim in claim_set.claims)


def test_unknown_evidence_ref_rejected() -> None:
    proposal = _sample_proposal()
    mutated = proposal.model_copy(
        update={
            "claim_proposals": (
                ClaimProposal(
                    statement="bad refs",
                    claim_kind=str(DIAGNOSIS_KIND),
                    supporting_evidence_ids=("evidence.unknown.node",),
                ),
            )
        }
    )
    with pytest.raises(ReasoningProposalValidationError, match="unknown evidence"):
        validate_reasoning_proposal(mutated, evidence_nodes=())


def test_critic_apply_resolutions_rejects_model_self_approval() -> None:
    claim_set = EvidenceClaimSet(
        claims=(
            convert_proposal_to_pending_claims(_sample_proposal(), prior_claim_set=None, critic_feedback=None)
            .claims[0]
            .model_copy(update={"resolution": ClaimResolution.SUPPORTED}),
        ),
        challenges=(),
    )
    with pytest.raises(ValueError, match="model_self_approved"):
        apply_critic_claim_resolutions(claim_set, {"evidence_nodes": []})


def test_validation_does_not_call_derive_hypothesis_dispositions() -> None:
    source = inspect.getsource(validate_claim_set_against_observations)
    assert "derive_hypothesis_dispositions" not in source


def test_investigator_canonical_path_has_no_derive_hypothesis_dispositions() -> None:
    from platform_proofs.scenarios.ai_incident_investigation import investigator_agent as mod

    source = inspect.getsource(mod.IncidentInvestigatorAgent.run_step)
    assert "derive_hypothesis_dispositions" not in source
    assert "_PRIOR_EVIDENCE_BY_RUN" not in source
    assert "_build_claims_from_assessment" not in source


def test_revision_claim_uses_application_supersedes_lineage() -> None:
    initial = convert_proposal_to_pending_claims(_sample_proposal(), prior_claim_set=None, critic_feedback=None)
    proposal = _sample_proposal().model_copy(
        update={
            "claim_proposals": (
                ClaimProposal(
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
        prior_claim_set=initial,
        critic_feedback=["unsupported inference"],
    )
    revised_claim = next(claim for claim in revised.claims if claim.claim_id == REVISED_CLAIM_ID)
    assert revised_claim.supersedes_claim_id == str(INITIAL_CLAIM_ID)


@pytest.mark.asyncio
async def test_application_survives_without_proof_evaluator() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.scenario import (
        OUTCOME_RESOLVED,
        build_runtime_bundle,
        execute_resolved_skeleton,
    )

    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert result.critic_verdict_passed
    assert result.claim_set
    assert result.evidence_nodes
