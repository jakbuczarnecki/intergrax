# © Artur Czarnecki. All rights reserved.

"""Deterministic domain evidence attribution for incident claim conversion (DS-E2E-12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.evidence_claims import EvidenceReferenceId, validate_evidence_reference_id
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import (
    IncidentEvidenceIds,
    IncidentObservations,
    comparison_weakens_overload,
    hypothesis_evidence_relations,
    telemetry_supports_degradation,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    HypothesisId,
)

_HYPOTHESIS_BY_ID: dict[Literal["H1", "H2", "H3"], HypothesisId] = {
    "H1": HypothesisId.H1,
    "H2": HypothesisId.H2,
    "H3": HypothesisId.H3,
}


@dataclass(frozen=True, slots=True)
class ClaimEvidenceAttribution:
    supporting_evidence_ids: tuple[EvidenceReferenceId, ...]
    contradicting_evidence_ids: tuple[EvidenceReferenceId, ...]


def _canonicalize_attribution(
    raw_support: tuple[str, ...],
    raw_contradict: tuple[str, ...],
    observable_ids: frozenset[str],
) -> ClaimEvidenceAttribution:
    support_seen: set[str] = set()
    support: list[EvidenceReferenceId] = []
    for item in raw_support:
        canonical = str(validate_evidence_reference_id(item))
        if canonical not in observable_ids or canonical in support_seen:
            continue
        support_seen.add(canonical)
        support.append(validate_evidence_reference_id(canonical))

    contradict_seen: set[str] = set()
    contradict: list[EvidenceReferenceId] = []
    for item in raw_contradict:
        canonical = str(validate_evidence_reference_id(item))
        if (
            canonical not in observable_ids
            or canonical in support_seen
            or canonical in contradict_seen
        ):
            continue
        contradict_seen.add(canonical)
        contradict.append(validate_evidence_reference_id(canonical))

    return ClaimEvidenceAttribution(
        supporting_evidence_ids=tuple(sorted(support, key=str)),
        contradicting_evidence_ids=tuple(sorted(contradict, key=str)),
    )


def _attribute_h3(
    observations: IncidentObservations,
    evidence_ids: IncidentEvidenceIds,
    observable_ids: frozenset[str],
) -> ClaimEvidenceAttribution:
    telemetry = observations.telemetry
    comparison = observations.comparison
    if (
        telemetry is not None
        and comparison is not None
        and telemetry_supports_degradation(telemetry)
        and comparison_weakens_overload(
            observations.workload,
            observations.throughput,
            comparison,
        )
    ):
        return _canonicalize_attribution(
            (evidence_ids.telemetry, evidence_ids.comparison),
            (),
            observable_ids,
        )
    raw_support, raw_contradict = hypothesis_evidence_relations(
        HypothesisId.H3,
        observations,
        evidence_ids,
    )
    return _canonicalize_attribution(raw_support, raw_contradict, observable_ids)


def attribute_claim_evidence(
    hypothesis_id: Literal["H1", "H2", "H3"],
    observations: IncidentObservations,
    evidence_ids: IncidentEvidenceIds,
    observable_ids: frozenset[str],
) -> ClaimEvidenceAttribution:
    """Bind support/contradiction evidence IDs from domain predicates — never from model output."""
    if hypothesis_id == "H3":
        return _attribute_h3(observations, evidence_ids, observable_ids)
    raw_support, raw_contradict = hypothesis_evidence_relations(
        _HYPOTHESIS_BY_ID[hypothesis_id],
        observations,
        evidence_ids,
    )
    return _canonicalize_attribution(raw_support, raw_contradict, observable_ids)
