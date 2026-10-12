# © Artur Czarnecki. All rights reserved.

"""Fail-closed PROD-Q-P0-R1-R1-R1 readiness gate (qualification harness only)."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_classification import classified_inventory
from tests.qualification.prod_q._p0_discovery import (
    anchor_resolution_failures,
    discovery_scan_failures,
    raw_discovery_duplicate_ids,
)
from tests.qualification.prod_q._p0_findings_register import known_prod_q_findings
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation, required_frz_ids
from tests.qualification.prod_q._p0_integrity import unclassified_indicator_ids
from tests.qualification.prod_q._p0_ownership import (
    alternate_composition_violations,
    auxiliary_na_without_symbol_evidence,
    build_ownership_reconciliation,
    ownership_soundness_violations,
    unresolved_role_classifications,
)
from tests.qualification.prod_q._p0_types import (
    CandidateInventoryStatus,
    MechanismRole,
    ProdQFindingDisposition,
    ProductionReadinessEvidenceState,
)

PROD_Q_P0_R1_R1_R1_STAGE_ID = "PROD-Q-P0-R1-R1-R1"


@lru_cache(maxsize=1)
def prod_q_p0_r1_r1_r1_gate_violations() -> tuple[str, ...]:
    violations: list[str] = []

    if discovery_scan_failures() or anchor_resolution_failures():
        violations.append("eligible scan/read/parse integrity != PASS")
    if raw_discovery_duplicate_ids():
        violations.append("duplicate candidate ids present")

    by_id = {r.candidate_id: r for r in classified_inventory()}
    unclassified = unclassified_indicator_ids(by_id)
    if unclassified:
        violations.append(f"unclassified indicators on current HEAD: {len(unclassified)}")

    for record in classified_inventory():
        if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY:
            if not any(ref.startswith("auxiliary_na:symbol:") for ref in record.current_head_evidence):
                violations.append(f"{record.candidate_id}: auxiliary N/A without symbol evidence")

    violations.extend(auxiliary_na_without_symbol_evidence())
    violations.extend(unresolved_role_classifications())
    violations.extend(ownership_soundness_violations())
    violations.extend(alternate_composition_violations())

    findings = known_prod_q_findings()
    if len(findings) != 8:
        violations.append(f"known findings register incomplete: {len(findings)}")
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
