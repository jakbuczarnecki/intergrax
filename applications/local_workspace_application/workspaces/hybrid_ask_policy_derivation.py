# © Artur Czarnecki. All rights reserved.

"""Bridge Tier-1 policy derivation contracts into Hybrid Ask planning."""

from __future__ import annotations

from local_workspace_application.workspaces.hybrid_ask_policy import (
    HybridAskPolicyError,
    IndexedEvidenceRequirementV1,
    LiveCallProposalV1,
    LiveEvidenceRequirementV1,
    RequiredEvidenceObligationV1,
    compose_evidence_obligations,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    DerivedEvidenceContractV1,
    DerivedEvidenceObligationV1,
    DerivedIndexedEvidenceObligationV1,
    DerivedLiveCallProposalV1,
    DerivedLiveEvidenceObligationV1,
    PolicyEvidenceBasisV1,
)


def map_derived_obligation(
    obligation: DerivedEvidenceObligationV1,
) -> RequiredEvidenceObligationV1:
    if isinstance(obligation, DerivedIndexedEvidenceObligationV1):
        return IndexedEvidenceRequirementV1(
            requirement_id=obligation.requirement_id,
            semantic_role=obligation.semantic_role,
            indexed_source_binding_id=obligation.indexed_source_binding_id,
            policy_origin=obligation.origin,
        )
    if isinstance(obligation, DerivedLiveEvidenceObligationV1):
        return LiveEvidenceRequirementV1(
            requirement_id=obligation.requirement_id,
            semantic_role=obligation.semantic_role,
            call_id=obligation.call_id,
            policy_origin=obligation.origin,
        )
    raise HybridAskPolicyError("derived_obligation_kind_unsupported")


def map_derived_live_call_proposal(
    proposal: DerivedLiveCallProposalV1,
) -> LiveCallProposalV1:
    return LiveCallProposalV1(
        call_id=proposal.call_id,
        live_access_binding_id=proposal.live_access_binding_id,
        capability_id=proposal.capability_id,
        typed_capability_request={
            entry.key: entry.value for entry in proposal.typed_capability_request
        },
    )


def map_derived_evidence_contract(
    contract: DerivedEvidenceContractV1,
) -> tuple[
    tuple[LiveCallProposalV1, ...],
    tuple[RequiredEvidenceObligationV1, ...],
    PolicyEvidenceBasisV1 | None,
]:
    return (
        tuple(
            map_derived_live_call_proposal(proposal)
            for proposal in contract.derived_live_call_proposals
        ),
        tuple(
            map_derived_obligation(obligation)
            for obligation in contract.derived_obligations
        ),
        contract.policy_basis,
    )


def merge_live_call_proposals(
    *,
    authoritative: tuple[LiveCallProposalV1, ...],
    additional: tuple[LiveCallProposalV1, ...],
) -> tuple[LiveCallProposalV1, ...]:
    """Additive live-call composition; duplicate call_id fails closed."""
    seen: set[str] = set()
    merged: list[LiveCallProposalV1] = []
    for proposal in (*authoritative, *additional):
        if proposal.call_id in seen:
            raise HybridAskPolicyError("duplicate_call_id")
        seen.add(proposal.call_id)
        merged.append(proposal)
    return tuple(merged)


def compose_authoritative_evidence_obligations(
    *,
    product: tuple[RequiredEvidenceObligationV1, ...],
    policy_derived: tuple[RequiredEvidenceObligationV1, ...],
    provider: tuple[RequiredEvidenceObligationV1, ...],
    caller_additive: tuple[RequiredEvidenceObligationV1, ...],
) -> tuple[RequiredEvidenceObligationV1, ...]:
    """Canonical authority order: product → policy → provider → caller additive."""
    return compose_evidence_obligations(
        authoritative=compose_evidence_obligations(
            authoritative=compose_evidence_obligations(
                authoritative=product,
                additional=policy_derived,
            ),
            additional=provider,
        ),
        additional=caller_additive,
    )
