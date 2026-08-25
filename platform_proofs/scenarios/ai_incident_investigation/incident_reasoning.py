# © Artur Czarnecki. All rights reserved.

"""Model-owned incident reasoning DTOs, validation, and claim conversion (APP-2BC)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
    mint_evidence_claim_id,
    validate_claim_kind,
    validate_evidence_reference_id,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from platform_proofs.scenarios.ai_incident_investigation.fixtures import HypothesisId
from platform_proofs.scenarios.ai_incident_investigation.investigation_observability import (
    IncidentClaimProposedDiagV1,
    IncidentClaimRevisedDiagV1,
    IncidentCompletionIntentDiagV1,
    IncidentEvidenceGapDiagV1,
    IncidentReasoningUpdateDiagV1,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
    DIAGNOSIS_KIND,
    INCIDENT_EVIDENCE_IDS,
)

LEGAL_HYPOTHESIS_IDS: frozenset[str] = frozenset({"H1", "H2", "H3"})
FORBIDDEN_MODEL_RESOLUTIONS: frozenset[ClaimResolution] = frozenset(
    {
        ClaimResolution.SUPPORTED,
        ClaimResolution.REJECTED,
        ClaimResolution.SUPERSEDED,
        ClaimResolution.INSUFFICIENT_EVIDENCE,
    }
)


class HypothesisDisposition(StrEnum):
    """Provisional model-owned hypothesis state — not authoritative claim resolution."""

    PLAUSIBLE = "plausible"
    WEAKENED = "weakened"
    REJECTED = "rejected"
    SUPPORTED = "supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SUPERSEDED = "superseded"
    PENDING = "pending"


class CompletionIntent(StrEnum):
    SUPPORTED_DIAGNOSIS = COMPLETION_SUPPORTED_DIAGNOSIS
    UNRESOLVED = COMPLETION_UNRESOLVED
    NEED_MORE_EVIDENCE = "need_more_evidence"


class HypothesisProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: Literal["H1", "H2", "H3"]
    disposition: HypothesisDisposition
    summary: str = Field(min_length=1, max_length=1024)
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    uncertainty: str = Field(default="", max_length=512)


class ClaimProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: Literal["H1", "H2", "H3"]
    statement: str = Field(min_length=1, max_length=4096)
    claim_kind: str = Field(min_length=1, max_length=128)
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    rationale: str = Field(default="", max_length=1024)
    replaces_prior_claim: bool = False


class IncidentReasoningProposal(BaseModel):
    """Scenario-local structured model output — provisional investigation reasoning."""

    model_config = ConfigDict(extra="forbid")

    hypotheses: tuple[HypothesisProposal, ...]
    preferred_hypothesis_id: Literal["H1", "H2", "H3"]
    uncertainty_class: str = Field(max_length=256)
    information_gaps: tuple[str, ...] = ()
    claim_proposals: tuple[ClaimProposal, ...]
    completion_intent: CompletionIntent
    action_objective: str = Field(max_length=512)
    follow_up_objective: str | None = Field(default=None, max_length=512)
    unresolved_reason: str | None = Field(default=None, max_length=1024)

    @field_validator("hypotheses")
    @classmethod
    def _validate_hypothesis_ids(
        cls, value: tuple[HypothesisProposal, ...]
    ) -> tuple[HypothesisProposal, ...]:
        seen: set[str] = set()
        for item in value:
            if item.hypothesis_id in seen:
                raise ValueError("duplicate hypothesis_id in proposal")
            seen.add(item.hypothesis_id)
            if item.hypothesis_id not in LEGAL_HYPOTHESIS_IDS:
                raise ValueError("illegal hypothesis_id")
        return value

    @model_validator(mode="after")
    def _validate_unresolved_intent(self) -> IncidentReasoningProposal:
        if self.completion_intent is CompletionIntent.UNRESOLVED and not (
            self.unresolved_reason and self.unresolved_reason.strip()
        ):
            raise ValueError("unresolved intent requires explicit unresolved_reason")
        if self.completion_intent is CompletionIntent.UNRESOLVED and not self.information_gaps:
            raise ValueError("unresolved intent requires decisive information_gaps")
        return self


class ClaimHypothesisBinding(BaseModel):
    """Scenario-local semantic binding — claim identity is independent of hypothesis."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    hypothesis_id: Literal["H1", "H2", "H3"]


@dataclass(frozen=True, slots=True)
class PendingClaimsConversion:
    claim_set: EvidenceClaimSet
    bindings: tuple[ClaimHypothesisBinding, ...]


@dataclass(frozen=True, slots=True)
class PriorInvestigationState:
    evidence_nodes: tuple[dict[str, object], ...]
    reasoning_proposal: IncidentReasoningProposal | None
    claim_set: EvidenceClaimSet | None
    claim_hypothesis_bindings: tuple[ClaimHypothesisBinding, ...]
    completion_intent: CompletionIntent | None
    summary: str


class ReasoningProposalValidationError(ValueError):
    """Malformed or inadmissible model reasoning output."""


def _known_evidence_ids(nodes: Sequence[dict[str, object]]) -> frozenset[str]:
    return frozenset(
        str(node["evidence_id"]) for node in nodes if node.get("evidence_id")
    )


def validate_reasoning_proposal(
    proposal: IncidentReasoningProposal,
    *,
    evidence_nodes: Sequence[dict[str, object]],
) -> None:
    known = _known_evidence_ids(evidence_nodes)
    if proposal.preferred_hypothesis_id not in LEGAL_HYPOTHESIS_IDS:
        raise ReasoningProposalValidationError("illegal preferred_hypothesis_id")

    for hypothesis in proposal.hypotheses:
        for evidence_id in (
            *hypothesis.supporting_evidence_ids,
            *hypothesis.contradicting_evidence_ids,
        ):
            if str(evidence_id) not in known:
                raise ReasoningProposalValidationError(
                    f"unknown evidence reference in hypothesis: {evidence_id}"
                )

    if not proposal.claim_proposals:
        raise ReasoningProposalValidationError("claim_proposals must be non-empty")

    for claim in proposal.claim_proposals:
        if not claim.statement.strip():
            raise ReasoningProposalValidationError("claim statement must be non-empty")
        validate_claim_kind(claim.claim_kind)
        for evidence_id in (
            *claim.supporting_evidence_ids,
            *claim.contradicting_evidence_ids,
        ):
            if str(evidence_id) not in known:
                raise ReasoningProposalValidationError(
                    f"unknown evidence reference in claim: {evidence_id}"
                )


def parse_claim_hypothesis_bindings(
    raw_bindings: object,
) -> tuple[ClaimHypothesisBinding, ...]:
    if not isinstance(raw_bindings, (list, tuple)):
        return ()
    bindings: list[ClaimHypothesisBinding] = []
    for item in raw_bindings:
        if isinstance(item, dict):
            bindings.append(ClaimHypothesisBinding.model_validate(item))
    return tuple(bindings)


def bindings_for_claim_set(
    claim_set: EvidenceClaimSet,
    bindings: Sequence[ClaimHypothesisBinding],
) -> dict[str, Literal["H1", "H2", "H3"]]:
    by_claim_id = {binding.claim_id: binding.hypothesis_id for binding in bindings}
    return {
        str(claim.claim_id): by_claim_id[str(claim.claim_id)]
        for claim in claim_set.claims
        if str(claim.claim_id) in by_claim_id
    }


def claim_id_for_hypothesis(
    bindings: Sequence[ClaimHypothesisBinding],
    hypothesis_id: Literal["H1", "H2", "H3"],
) -> str | None:
    for binding in bindings:
        if binding.hypothesis_id == hypothesis_id:
            return binding.claim_id
    return None


def convert_proposal_to_pending_claims(
    proposal: IncidentReasoningProposal,
    *,
    prior_claim_set: EvidenceClaimSet | None,
    prior_bindings: Sequence[ClaimHypothesisBinding] = (),
    critic_feedback: Sequence[str] | None,
) -> PendingClaimsConversion:
    prior_by_hypothesis: dict[str, str] = {
        binding.hypothesis_id: binding.claim_id for binding in prior_bindings
    }

    claims: list[EvidenceBackedClaim] = []
    bindings: list[ClaimHypothesisBinding] = []
    for claim_proposal in proposal.claim_proposals:
        if claim_proposal.hypothesis_id not in LEGAL_HYPOTHESIS_IDS:
            raise ReasoningProposalValidationError("illegal hypothesis_id on claim proposal")
        supersedes: str | None = None
        if claim_proposal.replaces_prior_claim:
            supersedes = prior_by_hypothesis.get(claim_proposal.hypothesis_id)
        claim_id = mint_evidence_claim_id()
        claims.append(
            EvidenceBackedClaim(
                claim_id=claim_id,
                statement=claim_proposal.statement,
                claim_kind=validate_claim_kind(claim_proposal.claim_kind),
                supporting_evidence_ids=tuple(
                    validate_evidence_reference_id(eid)
                    for eid in claim_proposal.supporting_evidence_ids
                ),
                contradicting_evidence_ids=tuple(
                    validate_evidence_reference_id(eid)
                    for eid in claim_proposal.contradicting_evidence_ids
                ),
                resolution=ClaimResolution.PENDING,
                supersedes_claim_id=supersedes,
            )
        )
        bindings.append(
            ClaimHypothesisBinding(
                claim_id=str(claim_id),
                hypothesis_id=claim_proposal.hypothesis_id,
            )
        )

    merged_bindings: list[ClaimHypothesisBinding] = list(prior_bindings)
    for binding in bindings:
        merged_bindings = [
            item for item in merged_bindings if item.hypothesis_id != binding.hypothesis_id
        ]
        merged_bindings.append(binding)

    if prior_claim_set is not None:
        new_ids = {claim.claim_id for claim in claims}
        for prior in prior_claim_set.claims:
            if prior.claim_id not in new_ids:
                claims.insert(0, prior)

    return PendingClaimsConversion(
        claim_set=EvidenceClaimSet(claims=tuple(claims), challenges=()),
        bindings=tuple(merged_bindings),
    )


def _prior_outputs_from_context(metadata: dict[str, object]) -> dict[str, Any]:
    raw = metadata.get("prior_agent_outputs")
    if isinstance(raw, dict):
        return raw
    return {}


def extract_prior_investigation_state(
    metadata: dict[str, object],
    *,
    node_id: str | None,
) -> PriorInvestigationState:
    prior_outputs = _prior_outputs_from_context(metadata)
    selected: dict[str, Any] | None = None
    if node_id and node_id in prior_outputs:
        selected = prior_outputs[node_id]
    elif prior_outputs:
        selected = next(iter(prior_outputs.values()))

    if not selected:
        return PriorInvestigationState(
            evidence_nodes=(),
            reasoning_proposal=None,
            claim_set=None,
            claim_hypothesis_bindings=(),
            completion_intent=None,
            summary="",
        )

    structured = selected.get("structured_data")
    if not isinstance(structured, dict):
        structured = selected

    domain_summary = structured.get("domain_summary")
    if not isinstance(domain_summary, dict):
        domain_summary = structured

    raw_nodes = domain_summary.get("evidence_nodes", [])
    evidence_nodes = tuple(
        dict(item) for item in raw_nodes if isinstance(item, dict)
    ) if isinstance(raw_nodes, list) else ()

    raw_proposal = domain_summary.get("reasoning_proposal")
    reasoning_proposal = (
        IncidentReasoningProposal.model_validate(raw_proposal)
        if isinstance(raw_proposal, dict)
        else None
    )

    raw_claim_set = domain_summary.get("claim_set")
    claim_set = (
        EvidenceClaimSet.model_validate(raw_claim_set)
        if isinstance(raw_claim_set, dict)
        else None
    )

    raw_intent = domain_summary.get("completion_mode")
    completion_intent = (
        CompletionIntent(str(raw_intent))
        if raw_intent in {item.value for item in CompletionIntent}
        else None
    )

    raw_bindings = domain_summary.get("claim_hypothesis_bindings")
    claim_hypothesis_bindings = parse_claim_hypothesis_bindings(raw_bindings)

    return PriorInvestigationState(
        evidence_nodes=evidence_nodes,
        reasoning_proposal=reasoning_proposal,
        claim_set=claim_set,
        claim_hypothesis_bindings=claim_hypothesis_bindings,
        completion_intent=completion_intent,
        summary=str(selected.get("summary") or domain_summary.get("summary") or ""),
    )


def build_reasoning_messages(
    *,
    evidence_nodes: Sequence[dict[str, object]],
    prior_state: PriorInvestigationState,
    critic_feedback: Sequence[str] | None,
    is_revision: bool,
) -> list[ChatMessage]:
    lines = [
        "Investigate Line 4 target attainment degradation using gathered evidence only.",
        "Compare competing hypotheses H1 sustained overload, H2 understaffing, H3 equipment degradation.",
        "Raw evidence acquisition tools and deterministic domain analysis tools are available.",
        "Use analysis tools when bounded deterministic comparison improves confidence.",
        "Do not treat workload-throughput correlation as causation.",
        "Propose only evidence-backed claims; cite evidence IDs from observations.",
        "Do not output claim_id, resolution, or supersedes_claim_id.",
        f"Investigation phase: {'revision' if is_revision else 'initial'}",
    ]
    if evidence_nodes:
        lines.append("Gathered evidence IDs:")
        for node in evidence_nodes:
            evidence_id = node.get("evidence_id")
            if not evidence_id:
                continue
            payload = node.get("payload")
            if isinstance(payload, dict) and payload.get("availability"):
                lines.append(f"- {evidence_id} availability={payload['availability']}")
            else:
                lines.append(f"- {evidence_id}")
    if prior_state.reasoning_proposal is not None:
        lines.append("Prior hypothesis summaries:")
        for hypothesis in prior_state.reasoning_proposal.hypotheses:
            lines.append(f"- {hypothesis.hypothesis_id}: {hypothesis.summary}")
    if critic_feedback:
        lines.append("Critic feedback requiring incremental correction:")
        lines.extend(f"- {item}" for item in critic_feedback)
    if is_revision:
        lines.append(
            "Perform incremental correction using all prior evidence; do not discard valid prior observations."
        )
    return [
        ChatMessage(role="system", content="\n".join(lines)),
        ChatMessage(role="user", content="Produce structured incident reasoning proposal."),
    ]


def propose_incident_reasoning(
    *,
    runtime_state: RuntimeState,
    evidence_nodes: Sequence[dict[str, object]],
    prior_state: PriorInvestigationState,
    critic_feedback: Sequence[str] | None,
    is_revision: bool,
) -> IncidentReasoningProposal:
    llm = runtime_state.context.config.llm_adapter
    if llm is None:
        raise RuntimeError("incident_reasoning_llm_missing")

    messages = build_reasoning_messages(
        evidence_nodes=evidence_nodes,
        prior_state=prior_state,
        critic_feedback=critic_feedback,
        is_revision=is_revision,
    )
    structured = llm.generate_structured(
        messages,
        IncidentReasoningProposal,
        temperature=0.0,
        run_id=runtime_state.run_id,
    )
    proposal = structured.parsed
    validate_reasoning_proposal(proposal, evidence_nodes=evidence_nodes)
    return proposal


def emit_reasoning_observability(
    *,
    runtime_state: RuntimeState,
    proposal: IncidentReasoningProposal,
    claim_set: EvidenceClaimSet,
    bindings: Sequence[ClaimHypothesisBinding],
    is_revision: bool,
    critic_feedback: Sequence[str] | None,
) -> None:
    binding_by_claim = {binding.claim_id: binding.hypothesis_id for binding in bindings}
    runtime_state.trace_event(
        component=TraceComponent.PLANNER,
        step="incident_reasoning_update",
        message="Incident investigator reasoning update",
        level=TraceLevel.INFO,
        payload=IncidentReasoningUpdateDiagV1(
            investigation_phase="revision" if is_revision else "initial",
            preferred_hypothesis_id=proposal.preferred_hypothesis_id,
            uncertainty_class=proposal.uncertainty_class,
            hypothesis_count=len(proposal.hypotheses),
            information_gap_count=len(proposal.information_gaps),
        ),
    )
    for gap in proposal.information_gaps:
        runtime_state.trace_event(
            component=TraceComponent.PLANNER,
            step="incident_evidence_gap",
            message="Incident investigator evidence gap",
            level=TraceLevel.INFO,
            payload=IncidentEvidenceGapDiagV1(
                investigation_phase="revision" if is_revision else "initial",
                information_gap=gap,
            ),
        )
    for claim in claim_set.claims:
        hypothesis_id = binding_by_claim.get(str(claim.claim_id))
        payload = (
            IncidentClaimRevisedDiagV1(
                claim_id=str(claim.claim_id),
                hypothesis_id=hypothesis_id,
                statement=claim.statement,
                critic_feedback=tuple(critic_feedback or ()),
            )
            if is_revision
            else IncidentClaimProposedDiagV1(
                claim_id=str(claim.claim_id),
                hypothesis_id=hypothesis_id,
                statement=claim.statement,
                supporting_evidence_count=len(claim.supporting_evidence_ids),
            )
        )
        runtime_state.trace_event(
            component=TraceComponent.PLANNER,
            step="incident_claim_revised" if is_revision else "incident_claim_proposed",
            message="Incident investigator claim update",
            level=TraceLevel.INFO,
            payload=payload,
        )
    runtime_state.trace_event(
        component=TraceComponent.PLANNER,
        step="incident_completion_intent",
        message="Incident investigator completion intent",
        level=TraceLevel.INFO,
        payload=IncidentCompletionIntentDiagV1(
            completion_intent=proposal.completion_intent.value,
            unresolved_reason=proposal.unresolved_reason,
        ),
    )


def completion_mode_from_proposal(proposal: IncidentReasoningProposal) -> str:
    if proposal.completion_intent is CompletionIntent.UNRESOLVED:
        return COMPLETION_UNRESOLVED
    return COMPLETION_SUPPORTED_DIAGNOSIS


def build_investigation_summary(proposal: IncidentReasoningProposal, *, is_revision: bool) -> str:
    if proposal.completion_intent is CompletionIntent.UNRESOLVED:
        if proposal.unresolved_reason:
            return (
                "Investigation remains unresolved: workload-only and staffing explanations "
                "are not supported, while the equipment hypothesis cannot be accepted "
                "because decisive telemetry for the incident window is unavailable."
            )
    preferred = next(
        (item for item in proposal.hypotheses if item.hypothesis_id == proposal.preferred_hypothesis_id),
        None,
    )
    base = preferred.summary if preferred is not None else proposal.action_objective
    if is_revision:
        return f"revised: {base}"
    return base
