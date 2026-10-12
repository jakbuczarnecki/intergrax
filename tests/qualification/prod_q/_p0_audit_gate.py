# © Artur Czarnecki. All rights reserved.

"""Fail-closed PROD-Q-P0 readiness gates (qualification harness only)."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_classification import classified_inventory
from tests.qualification.prod_q._p0_discovery import (
    anchor_resolution_failures,
    discovery_scan_failures,
    raw_discovery_duplicate_ids,
)
from tests.qualification.prod_q._p0_findings_register import (
    concern_finding_reconciliation_violations,
    factory_ownership_resolved_without_proof_violations,
    historical_prod_q_findings,
    known_prod_q_findings,
)
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation, required_frz_ids
from tests.qualification.prod_q._p0_integrity import unclassified_indicator_ids
from tests.qualification.prod_q._p0_ownership import (
    alternate_composition_violations,
    auxiliary_evidence_is_sufficient,
    auxiliary_na_without_symbol_evidence,
    auxiliary_verified_without_relationship_proof_violations,
    build_auxiliary_reconciliation_metrics,
    build_ownership_reconciliation,
    ownership_soundness_violations,
    unresolved_role_classifications,
)
from tests.qualification.prod_q._p0_types import (
    MechanismRole,
    ProductionReadinessClassification,
    ProductionReadinessEvidenceState,
)

PROD_Q_P0_R1_R1_R1_STAGE_ID = "PROD-Q-P0-R1-R1-R1"
PROD_Q_P0_R1_R1_R1_R1_STAGE_ID = "PROD-Q-P0-R1-R1-R1-R1"
PROD_Q_P0_R1_R1_R1_R1_R1_STAGE_ID = "PROD-Q-P0-R1-R1-R1-R1-R1"
PROD_Q_P0_R1_R1_R1_R1_R1_R1_STAGE_ID = "PROD-Q-P0-R1-R1-R1-R1-R1-R1"


@lru_cache(maxsize=1)
def prod_q_p0_r1_r1_r1_gate_violations() -> tuple[str, ...]:
    return prod_q_p0_r1_r1_r1_r1_r1_r1_gate_violations()


def fake_auxiliary_na_violations(
    inventory: tuple[ProductionReadinessClassification, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{record.candidate_id}: fake auxiliary_na without evidence"
        for record in inventory
        if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY
        and any(ref.startswith("auxiliary_na:symbol:") for ref in record.current_head_evidence)
        and not any(ref.startswith("auxiliary_evidence:") for ref in record.current_head_evidence)
    )


def prod_q_p0_inventory_reconciliation_violations(
    inventory: tuple[ProductionReadinessClassification, ...],
) -> tuple[str, ...]:
    from tests.qualification.prod_q._p0_ownership import _reconciliation_from

    violations: list[str] = []
    by_id = {r.candidate_id: r for r in inventory}
    unclassified = unclassified_indicator_ids(by_id)
    if unclassified:
        violations.append(f"unclassified indicators on current HEAD: {len(unclassified)}")

    violations.extend(auxiliary_na_without_symbol_evidence(inventory))
    violations.extend(auxiliary_verified_without_relationship_proof_violations(inventory))
    violations.extend(fake_auxiliary_na_violations(inventory))
    violations.extend(unresolved_role_classifications(inventory))
    violations.extend(ownership_soundness_violations(inventory))
    violations.extend(alternate_composition_violations(inventory))
    violations.extend(concern_finding_reconciliation_violations(inventory))

    aux_metrics = build_auxiliary_reconciliation_metrics(inventory)
    if aux_metrics.auxiliary_discovered != (
        aux_metrics.auxiliary_justified + aux_metrics.auxiliary_unresolved
    ):
        violations.append("auxiliary reconciliation identity does not balance")
    unverified_auxiliary_na = sum(
        1
        for record in inventory
        if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY
        and record.evidence_state == ProductionReadinessEvidenceState.N_A_WITH_EVIDENCE
        and not auxiliary_evidence_is_sufficient(record)
    )
    if unverified_auxiliary_na > 0:
        violations.append(f"unverified auxiliary N/A: {unverified_auxiliary_na}")
    if aux_metrics.auxiliary_unresolved != 0:
        violations.append(f"auxiliary unresolved: {aux_metrics.auxiliary_unresolved}")
    if aux_metrics.auxiliary_falsely_excluded != 0:
        violations.append(f"auxiliary falsely excluded: {aux_metrics.auxiliary_falsely_excluded}")

    recon = _reconciliation_from(inventory)
    accounted = (
        recon.semantic_mechanism_count
        + recon.auxiliary_discovery_count
        + recon.unresolved_classification_count
    )
    if accounted != recon.candidate_count:
        violations.append("role classification counters do not cover all candidates")
    ownership_accounted = (
        recon.ownership_resolved
        + recon.ownership_unresolved
        + recon.ownership_ambiguous
        + recon.ownership_not_applicable_with_evidence
        + recon.unresolved_classification_count
    )
    if ownership_accounted != recon.candidate_count:
        violations.append("ownership reconciliation counters do not cover all candidates")
    return tuple(violations)


@lru_cache(maxsize=1)
def prod_q_p0_r1_r1_r1_r1_gate_violations() -> tuple[str, ...]:
    return prod_q_p0_r1_r1_r1_r1_r1_r1_gate_violations()


@lru_cache(maxsize=1)
def prod_q_p0_r1_r1_r1_r1_r1_r1_gate_violations() -> tuple[str, ...]:
    violations: list[str] = list(prod_q_p0_r1_r1_r1_r1_r1_gate_violations())
    violations.extend(factory_ownership_resolved_without_proof_violations())
    return tuple(violations)


@lru_cache(maxsize=1)
def prod_q_p0_r1_r1_r1_r1_r1_gate_violations() -> tuple[str, ...]:
    violations: list[str] = []

    if discovery_scan_failures() or anchor_resolution_failures():
        violations.append("eligible scan/read/parse integrity != PASS")
    if raw_discovery_duplicate_ids():
        violations.append("duplicate candidate ids present")

    inventory = classified_inventory()
    violations.extend(prod_q_p0_inventory_reconciliation_violations(inventory))

    historical = historical_prod_q_findings()
    if len(historical) != 8:
        violations.append(f"historical findings lineage incomplete: {len(historical)}")
    for item in historical:
        if not item.evidence_refs:
            violations.append(f"{item.finding_id}: missing evidence_refs")

    findings = known_prod_q_findings()
    if len(findings) < 8:
        violations.append(f"reconciled findings register incomplete: {len(findings)}")
    for item in findings:
        if not item.evidence_refs:
            violations.append(f"{item.finding_id}: missing evidence_refs")

    frz = build_frz_reconciliation()
    frz_ids = [r.criterion_id for r in frz]
    if frz_ids != list(required_frz_ids()) or len(frz_ids) != 33:
        violations.append("FRZ reconciliation not exactly-once for 33 criteria")
    for record in frz:
        if record.evidence_state == ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD:
            if not any("gate:" in ref for ref in record.evidence_refs):
                violations.append(f"{record.criterion_id}: PROVEN without gate evidence")

    recon = build_ownership_reconciliation()
    accounted = (
        recon.semantic_mechanism_count
        + recon.auxiliary_discovery_count
        + recon.unresolved_classification_count
    )
    if accounted != recon.candidate_count:
        violations.append("role classification counters do not cover all candidates")
    ownership_accounted = (
        recon.ownership_resolved
        + recon.ownership_unresolved
        + recon.ownership_ambiguous
        + recon.ownership_not_applicable_with_evidence
        + recon.unresolved_classification_count
    )
    if ownership_accounted != recon.candidate_count:
        violations.append("ownership reconciliation counters do not cover all candidates")

    return tuple(violations)


def prod_q_p0_r1_r1_r1_ready_for_audit() -> bool:
    return prod_q_p0_r1_r1_r1_gate_violations() == ()


def prod_q_p0_r1_r1_r1_r1_ready_for_audit() -> bool:
    return prod_q_p0_r1_r1_r1_r1_gate_violations() == ()


def prod_q_p0_r1_r1_r1_r1_r1_ready_for_audit() -> bool:
    return prod_q_p0_r1_r1_r1_r1_r1_gate_violations() == ()


def prod_q_p0_r1_r1_r1_r1_r1_r1_ready_for_audit() -> bool:
    return prod_q_p0_r1_r1_r1_r1_r1_r1_gate_violations() == ()
