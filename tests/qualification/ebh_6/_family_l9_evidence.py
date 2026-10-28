# © Artur Czarnecki. All rights reserved.

"""Per-family L9 evidence bundles — implementer SSOT (Protocol 03 remediation R2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_6._family_catalog import ImplementerProposedDisposition


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


class FamilyL9EvidenceDisposition(StrEnum):
    PASS_CANDIDATE = "PASS_CANDIDATE"
    SCOPED_CANDIDATE = "SCOPED CANDIDATE"
    EVIDENCE_REQUIRED = "EVIDENCE REQUIRED"


@dataclass(frozen=True, slots=True)
class FamilyL9EvidenceBundle:
    family_id: str
    frz_criteria: tuple[str, ...]
    sys_inv_refs: tuple[str, ...]
    invariant_under_test: str
    gate_test_ids: tuple[str, ...]
    qual_doc_pointer: str
    implementer_proposed_disposition: ImplementerProposedDisposition
    blocker_id: str | None
    scope_limitation: str
    l9_evidence_disposition: FamilyL9EvidenceDisposition
    gate_proves: str
    family_scope_covered: str
    reconciliation_blocker: str | None = None


_EE_A1 = "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py"
_EE_OWN = "tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py"
_GOV_FREEZE = "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py"
_REC_PLANE = "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
_REC_W3 = "tests/unit/runtime/architecture/test_w3_c4_decision_durable_recovery_canonical_wiring.py"
_SEC_R1 = "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py"
_REL_CAP = "tests/unit/runtime/architecture/test_ee_b1_2_capacity_architecture_gate.py"
_TEN = "tests/unit/runtime/security/test_tenant_scope.py"
_W5_B2 = "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py"
_BND = "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
_EBH6_REG = "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py"
_EBH6_MECH = "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py"


def _b(
    family_id: str,
    frz: tuple[str, ...],
    sys_inv: tuple[str, ...],
    invariant: str,
    gates: tuple[str, ...],
    qual_doc: str,
    impl: ImplementerProposedDisposition,
    blocker: str | None,
    scope_note: str,
    disp: FamilyL9EvidenceDisposition,
    proves: str,
    covered: str,
    recon: str | None = None,
) -> FamilyL9EvidenceBundle:
    return FamilyL9EvidenceBundle(
        family_id,
        frz,
        sys_inv,
        invariant,
        gates,
        qual_doc,
        impl,
        blocker,
        scope_note,
        disp,
        proves,
        covered,
        recon,
    )


FAMILY_L9_EVIDENCE: Final[tuple[FamilyL9EvidenceBundle, ...]] = (
    _b(
        "BND",
        ("FRZ-BND-01", "FRZ-BND-03", "FRZ-BND-05"),
        ("SYS-INV-BND-01", "SYS-INV-BND-02"),
        "Tier import directions and no reverse dependency on production paths.",
        (_nid(_BND, "test_hardening_3_contracts_do_not_import_runtime_except_allowlist"),),
        "EBH-2I / EBH-3 scoped historical + tier gate @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "Not global FRZ-BND PASS",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Static tier-0 import prohibition (intergrax/ must not import agents/applications).",
        "Tier-boundary slice only — not full composition reachability census.",
    ),
    _b(
        "OWN",
        ("FRZ-OWN-01", "FRZ-OWN-04", "FRZ-OWN-05"),
        ("SYS-INV-OWN-01",),
        "Exactly-one semantic owner on inventoried runtime/contract surfaces.",
        (_nid(_EE_OWN, "test_ee_final_arch_owner_matrix_documented"),),
        "EE final arch owner uniqueness @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        "EBH6-BLK-L9-OWN-01",
        "Closed-world rescan — not whole-platform ownership closure",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Documented owner matrix parity vs EE final arch facts module.",
        "Documented surfaces in EE final arch pack — not live duplicate-owner scan @ HEAD.",
    ),
    _b(
        "CTR",
        ("FRZ-CTR-02", "FRZ-CTR-06"),
        ("SYS-INV-CTR-01", "SYS-INV-VBN-01"),
        "No unresolved PROHIBITED contract leakage in register scope.",
        (_nid(_EBH6_REG, "test_ebh6_prohibited_unresolved_zero"),),
        "EBH-6 register + INT-CONFIG historical",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "Historical INT-CONFIG scoped",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Register mechanical count: prohibited unresolved = 0 in frozen EBH-6 scope.",
        "Findings register scope — not runtime contract leakage sweep.",
    ),
    _b(
        "TYP",
        ("FRZ-TYP-01", "FRZ-TYP-03"),
        ("SYS-INV-TYP-01",),
        "Tracked typing debt ratified or remediated on public seams.",
        (_nid(_EBH6_REG, "test_ebh6_tracked_debt_findings_ratified"),),
        "Q2-D2 ratified in findings register",
        ImplementerProposedDisposition.PASS_SCOPED,
        None,
        "Q2-D2 ADR — not global TYP PASS",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Register rows for Q2-D2 / BND-COMPAT-R3 are RATIFIED — not typing re-run.",
        "Register disposition only — not public-seam typing re-audit.",
    ),
    _b(
        "PLG",
        ("FRZ-PLG-01", "FRZ-PLG-04"),
        ("SYS-INV-PLG-01",),
        "Plugin SPI and provider selection without RAG-local registry.",
        ("tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete",),
        "EBH-5 inventory @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "EBH-5 scoped",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "EBH-5 exit inventory completeness (qual mechanical gate).",
        "Inventory completeness — not plugin SPI behavioral matrix @ HEAD.",
    ),
    _b(
        "RPL",
        ("FRZ-RPL-01", "FRZ-RPL-04"),
        ("SYS-INV-RPL-01",),
        "RPL-G regression and CONFIG-X blockers empty @ HEAD.",
        (
            "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py::"
            "test_ebh5_rpl_g_config_x_active_blockers_remain_empty",
        ),
        "EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX — 51 NYQ documented",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "51 NYQ — EBH-7 convergence",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "RPL-G CONFIG-X active blocker list empty @ HEAD.",
        "RPL-G slice — 51 matrix rows remain NOT YET QUALIFIED.",
    ),
    _b(
        "EXE",
        ("FRZ-EXE-*",),
        ("SYS-INV-EXE-01",),
        "Single ExecutionRuntime ownership — no shadow execution path on L3 samples.",
        (_nid(_EE_A1, "test_ee_a1_p0_inventory_reports_zero_production_bypass"),),
        "EE-A1 certification gate @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        "EBH6-BLK-L9-EXE-01",
        "L3 execution mechanisms cross-ref in _l3_reachability_evidence.py",
        FamilyL9EvidenceDisposition.PASS_CANDIDATE,
        "EE-A1 P0 inventory reports zero production execution bypass paths.",
        "EE-A1 certified surfaces — aligned with L3 execution mechanism samples.",
    ),
    _b(
        "GOV",
        ("FRZ-GOV-*", "FRZ-EXE-*"),
        ("SYS-INV-GOV-01", "SYS-INV-GOV-02"),
        "Governance does not execute effects; decision contracts fail-closed.",
        (
            _nid(_GOV_FREEZE, "test_npsc42_h1_execution_does_not_host_policy_engine"),
            _nid(_GOV_FREEZE, "test_npsc42_h1_governance_planes_do_not_mint_execution_identity"),
        ),
        "GOV-X2 historical + boundary freeze @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        "EBH6-BLK-L9-GOV-01",
        "Subset replay — not full GOV-X2 re-run",
        FamilyL9EvidenceDisposition.PASS_CANDIDATE,
        "NPSC42 H1 freeze: execution hosts no policy engine; governance planes do not mint execution identity.",
        "NPSC42 H1 frozen surface — not full GOV-X2 replay.",
    ),
    _b(
        "STA",
        ("FRZ-STA-*",),
        ("SYS-INV-STA-01",),
        "SQLite annex 38 touchpoints classified @ HEAD.",
        (_nid(_EBH6_MECH, "test_ebh6_exit_sqlite_annex_accountability"),),
        "STATE-X historical + EBH-5 annex",
        ImplementerProposedDisposition.PASS_SCOPED,
        None,
        "STATE-X scoped",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "SQLite annex paths accountable vs EBH-5 touchpoint registry.",
        "Annex accountability — not canonical store behavioral qualification.",
    ),
    _b(
        "REC",
        ("FRZ-REC-*",),
        ("SYS-INV-REC-01",),
        "Recovery plane ownership — no alternate recovery executor on L3 paths.",
        (
            _nid(_REC_PLANE, "test_no_recovery_god_object_in_production"),
            _nid(_REC_W3, "test_decision_durable_recovery_has_no_production_bypass"),
        ),
        "Guard matrix #5 recovery gates @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        "EBH6-BLK-L9-REC-01",
        "Drift replay — not DIST-SCALE-X full",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "No recovery god-object; decision durable recovery wiring has no production bypass.",
        "Selected recovery gates — not full DIST-SCALE-X / NPSC5e replay.",
    ),
    _b(
        "CMP",
        ("FRZ-CMP-*",),
        ("SYS-INV-CMP-01",),
        "COMPAT-X policy unchanged — no EBH-6 consumer blast-radius proof.",
        (),
        "COMPAT-X CLOSED @ historical SHA — EBH_6_SCOPE_RECONCILIATION_DOSSIER §CMP",
        ImplementerProposedDisposition.NA_WITH_EVIDENCE,
        None,
        "EVOL-X consumer impact out of scope — acceptable Path B",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Historical COMPAT-X closure preserved; EBH-6 records no consumer blast-radius regression in scope.",
        "Policy-unchanged attestation @ historical SHA — not HEAD behavioral replay.",
        None,
    ),
    _b(
        "CFG",
        ("FRZ-CFG-01", "FRZ-CFG-08"),
        ("SYS-INV-CFG-01",),
        "CONFIG-X 54 concerns reconciled; historical PASS preserved.",
        (
            "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py::"
            "test_ebh6_config_x_all_concerns_mapped_at_head",
        ),
        "CONFIG-X @ fac9a43f historical",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "FRZ-CFG PASS not rewritten",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "All CONFIG-X concerns mapped at HEAD (reconciliation gate).",
        "Mapping completeness — not live configuration materialization audit.",
    ),
    _b(
        "SEC",
        ("FRZ-SEC-*",),
        ("SYS-INV-SEC-01",),
        "Security composition fail-closed — external fail-open rejected @ HEAD.",
        (_nid(_SEC_R1, "test_r1_sec_01_canonical_composition_rejects_external_fail_open"),),
        "CTRL-X R1 @ HEAD replay",
        ImplementerProposedDisposition.REPLAY_PASS,
        "EBH6-BLK-L9-SEC-01",
        "CTRL-X historical SHA evidence preserved separately",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Canonical security composition rejects external fail-open middleware.",
        "CTRL-X R1 single composition scenario — not full SEC family.",
    ),
    _b(
        "REL",
        ("FRZ-REL-*",),
        ("SYS-INV-REL-01",),
        "Capacity preview ≠ execution owner (EE-B1.2).",
        (_nid(_REL_CAP, "test_ee_b1_2_capacity_contract_no_identity_mint"),),
        "HARNESS-W4 / EE-B1.2 guard @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        "EBH6-BLK-L9-REL-01",
        "Selected guard replay",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Capacity contract surfaces do not mint execution identity.",
        "EE-B1.2 capacity slice — not full reliability/backpressure matrix.",
    ),
    _b(
        "CTL",
        ("FRZ-CTL-*",),
        ("SYS-INV-CTL-01",),
        "Control-plane middleware fail-open rejected before attach.",
        (
            _nid(
                "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py",
                "test_r2_mw_05_fail_open_rejected_before_attach",
            ),
        ),
        "CTRL_X_ENTERPRISE_CONTROL_PLANE_RECERTIFICATION @ 318a4d83…",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "Drift-only",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Fail-open middleware rejected before attach (CTRL-X R2).",
        "Single middleware attach gate — not twelve-plane drift replay.",
    ),
    _b(
        "OBS",
        ("FRZ-OBS-*",),
        ("SYS-INV-OBS-01",),
        "Observability does not mint execution truth.",
        (_nid(_W5_B2, "test_production_composition_creates_bounded_sink"),),
        "HARNESS-W5 B2 wiring @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "TRACE-X global completeness not claimed",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Production composition creates bounded observability sink (non-authoritative export).",
        "W5-B2 wiring sample — not global observability qualification.",
    ),
    _b(
        "TRC",
        ("FRZ-TRC-*",),
        ("SYS-INV-TRC-01",),
        "Sanctioned composition roots registered for trace spine.",
        (_nid(_EBH6_MECH, "test_ebh6_exit_sanctioned_composition_roots_non_empty"),),
        "TRACE-X P4 registry parity",
        ImplementerProposedDisposition.PASS,
        None,
        "Scoped TRC contribution",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Sanctioned production composition root registry non-empty @ HEAD.",
        "Registry presence — not end-to-end trace spine behavioral proof.",
    ),
    _b(
        "PRD",
        ("FRZ-PRD-*",),
        (),
        "Architecture bypass census only — live PROD-Q slice excluded.",
        (),
        "PROD-Q Path B — EBH_6_SCOPE_RECONCILIATION_DOSSIER §PRD",
        ImplementerProposedDisposition.NA_WITH_EVIDENCE,
        None,
        "Live vendor qualification OPEN — acceptable exclusion",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Roadmap §3.0.8 Path B excludes live PROD-Q vendor slice from EBH-6 closure.",
        "Architecture census obligation met via closed-world register — not live provider qual.",
        None,
    ),
    _b(
        "REG",
        ("FRZ-REG-*",),
        (),
        "EBH-6 mechanical gates — not QUAL-X closure.",
        (
            _nid(_EBH6_MECH, "test_ebh6_exit_mandatory_docs_present"),
            _nid(_EBH6_MECH, "test_ebh6_exit_findings_register_complete"),
            _nid(_EBH6_REG, "test_ebh6_prohibited_unresolved_zero"),
            "tests/qualification/ebh_6/test_ebh_6_l3_and_family_evidence_integrity.py::test_aj_evidence_required_honest_not_zero",
        ),
        "QUAL-X remains OPEN — EBH-6 regression inventory @ HEAD",
        ImplementerProposedDisposition.PENDING_TO_REPLAY_PASS,
        None,
        "QUAL-X parent OPEN — EBH-6 scoped REG contribution only",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "EBH-6 qual gate bundle replays register + L3/family integrity — not global FRZ-REG.",
        "tests/qualification/ebh_6/* integrity + mechanical exit — not QUAL-X Population E.",
        None,
    ),
    _b(
        "HRN",
        ("FRZ-HRN-*",),
        ("SYS-INV-HRN-01",),
        "Harness W5 delivery stack non-authoritative for execution.",
        (_nid(_W5_B2, "test_published_event_reaches_bounded_sink"),),
        "HARNESS-W5 scoped @ HEAD",
        ImplementerProposedDisposition.REPLAY_PASS,
        "EBH6-BLK-L9-HRN-01",
        "W5/W6 subset only",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Published event reaches bounded sink — harness delivery non-authoritative path.",
        "W5-B2 subset — not full harness non-authority matrix.",
    ),
    _b(
        "TEN",
        ("FRZ-TEN-*",),
        ("SYS-INV-TEN-01",),
        "Tenant scope adversarial matrix on security boundary.",
        (_nid(_TEN, "test_tenant_scope_adversarial_matrix"),),
        "TENANT-X historical — no global TEN PASS",
        ImplementerProposedDisposition.REPLAY_PASS_SCOPED,
        "EBH6-BLK-L9-TEN-01",
        "Spot-check — not cross-subsystem TEN closure",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Tenant scope adversarial matrix on test_tenant_scope surfaces.",
        "Security module spot-check — not global tenant propagation closure.",
    ),
    _b(
        "DOC",
        ("FRZ-DOC-*",),
        (),
        "Mandatory EBH-6 qualification docs present @ HEAD.",
        (
            _nid(_EBH6_MECH, "test_ebh6_exit_mandatory_docs_present"),
            "tests/qualification/ebh_6/test_ebh_6_scope_reconciliation_integrity.py::test_ebh6_scope_dossier_content_controls",
        ),
        "EBH-6 dossier set + scope reconciliation dossier",
        ImplementerProposedDisposition.PASS,
        None,
        "FRZ-DOC global OPEN until ARCH-FREEZE",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Mandatory qual doc presence + scope dossier SHA/link/roadmap cross-check gate.",
        "Doc inventory + content controls — not FRZ-DOC-01..06 full content PASS.",
        None,
    ),
    _b(
        "VBN",
        ("FRZ-CTR-02", "FRZ-RPL-01"),
        ("SYS-INV-VBN-01",),
        "Hardcoding register mechanical exit @ HEAD.",
        (
            _nid(_EBH6_REG, "test_ebh6_findings_unclassified_zero"),
            _nid(_EBH6_REG, "test_ebh6_prohibited_unresolved_zero"),
        ),
        "CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT",
        ImplementerProposedDisposition.REPLAY_PASS,
        None,
        "Independent L9 still required",
        FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
        "Register unclassified=0 and prohibited unresolved=0 in frozen scope.",
        "Register mechanical exit — not runtime vendor construction rescan.",
    ),
)


@lru_cache(maxsize=1)
def family_l9_bundle_by_id() -> dict[str, FamilyL9EvidenceBundle]:
    return {row.family_id: row for row in FAMILY_L9_EVIDENCE}


def count_family_l9_dispositions() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in FAMILY_L9_EVIDENCE:
        key = row.l9_evidence_disposition.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def assert_family_l9_bundles_complete() -> None:
    assert len(FAMILY_L9_EVIDENCE) >= 23
    blockers = {
        "EBH6-BLK-L9-OWN-01",
        "EBH6-BLK-L9-EXE-01",
        "EBH6-BLK-L9-GOV-01",
        "EBH6-BLK-L9-REC-01",
        "EBH6-BLK-L9-SEC-01",
        "EBH6-BLK-L9-REL-01",
        "EBH6-BLK-L9-HRN-01",
        "EBH6-BLK-L9-TEN-01",
    }
    for bid in blockers:
        matches = [b for b in FAMILY_L9_EVIDENCE if b.blocker_id == bid]
        assert len(matches) == 1, bid
        assert matches[0].gate_test_ids, bid
        assert matches[0].l9_evidence_disposition != FamilyL9EvidenceDisposition.EVIDENCE_REQUIRED, bid


def assert_family_l9_reconciliation_complete() -> None:
    assert_family_l9_bundles_complete()
    for row in FAMILY_L9_EVIDENCE:
        assert row.gate_proves.strip(), row.family_id
        assert row.family_scope_covered.strip(), row.family_id
        if row.l9_evidence_disposition == FamilyL9EvidenceDisposition.EVIDENCE_REQUIRED:
            assert row.reconciliation_blocker, row.family_id
        if not row.gate_test_ids:
            assert row.l9_evidence_disposition in (
                FamilyL9EvidenceDisposition.EVIDENCE_REQUIRED,
                FamilyL9EvidenceDisposition.SCOPED_CANDIDATE,
            ), row.family_id
        for node in row.gate_test_ids:
            if "mandatory_docs" in node or "findings_unclassified" in node:
                assert row.l9_evidence_disposition != FamilyL9EvidenceDisposition.PASS_CANDIDATE, node
