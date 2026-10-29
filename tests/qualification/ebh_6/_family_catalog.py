# © Artur Czarnecki. All rights reserved.

"""EBH-6 architecture-family recertification catalog — mechanical SSOT for matrix rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.ebh_6._evidence_manifest import (
    CONFIG_X_HISTORICAL_PASS_SHA,
    EBH5_INDEPENDENT_AUDIT_SHA,
)


class FamilyEvidenceFreshness(StrEnum):
    HISTORICAL = "HISTORICAL @ scoped SHA"
    CURRENT = "CURRENT @ EBH-6 HEAD"
    MISSING = "MISSING — EBH-6 obligation"


class FamilyIndependentDisposition(StrEnum):
    PENDING = "PENDING — L9 Protocol 03"
    REPLAY_PASS = "REPLAY PASS (drift-only)"
    NOT_APPLICABLE_WITH_EVIDENCE = "N/A — WITH EVIDENCE"


class ImplementerProposedDisposition(StrEnum):
    PENDING = "PENDING (proposed)"
    REPLAY_PASS = "REPLAY PASS (proposed)"
    PASS = "PASS (proposed)"
    PASS_SCOPED = "PASS scoped (proposed)"
    REPLAY_PASS_SCOPED = "REPLAY PASS scoped (proposed)"
    NA_WITH_EVIDENCE = "N/A — WITH EVIDENCE (proposed)"
    PENDING_TO_REPLAY_PASS = "PENDING→REPLAY PASS (proposed)"


@dataclass(frozen=True, slots=True)
class ArchitectureFamilyRecertRow:
    family_id: str
    ecap_theme: str
    frz_family: str
    primary_historical_owner: str
    authority: str
    scope_paths: tuple[str, ...]
    existing_evidence_sha: str
    freshness: FamilyEvidenceFreshness
    missing_proof: str
    gate_test_ids: tuple[str, ...]
    remediation_batch: str
    acceptance_rule: str
    independent_disposition: FamilyIndependentDisposition
    implementer_proposed_disposition: ImplementerProposedDisposition
    notes: str = ""


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


_BND = "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
_EBH6_MECH = "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py"
_EBH6_REG = "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py"
_Q2D2_REMEDIATION_NID = _nid(
    "tests/qualification/ebh_6/test_ebh_6_q2d2_dual_path.py",
    "test_ebh6_q2d2_path_b_remediation_proves_registered_violation_removed",
)
_EBH6_CFG = "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py"
_EBH6_FAM = "tests/qualification/ebh_6/test_ebh_6_architecture_family_matrix_integrity.py"
_EBH5_MECH = "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py"
_EBH5_RPLG = "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py"
_CTRL_CATALOG = "tests/qualification/control_planes/ctrl_x/catalog.py"


_EBH6_FAMILY_CATALOG_BASE: tuple[ArchitectureFamilyRecertRow, ...] = (
    ArchitectureFamilyRecertRow(
        "BND",
        "Hard layer boundaries",
        "FRZ-BND-*",
        "EBH-2*, EBH-3, EBH-4",
        "Tier import gate + composition reachability",
        ("intergrax/", "agents/", "applications/"),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "",
        (_nid(_BND, "test_hardening_3_contracts_do_not_import_runtime_except_allowlist"),),
        "P1",
        "No new PROVEN boundary violation on production path",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.REPLAY_PASS,
        "DEBT-BND-COMPAT-R3 tracked lab-only in findings register",
    ),
    ArchitectureFamilyRecertRow(
        "OWN",
        "Exactly-one ownership",
        "FRZ-OWN-*",
        "EBH-2*, EBH-3",
        "INTERGRAX_ARCHITECTURE_PRINCIPLES ownership matrix",
        ("intergrax/runtime/", "intergrax/contracts/"),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "EBH6-BLK-L9-OWN-01 — EE final arch scoped replay (platform duplicate scan → DUP-X)",
        (
            _nid(
                "tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py",
                "test_ee_final_arch_owner_matrix_documented",
            ),
            _nid(
                "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py",
                "test_ee_a1_p0_inventory_reports_zero_production_bypass",
            ),
        ),
        "P1",
        "FRZ-OWN scoped @ EE-FINAL-ARCH + EE-A1 — not DUP-X closed-world duplicate scan",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "CTR",
        "Canonical contracts",
        "FRZ-CTR-*",
        "EBH-2*, INT-CONFIG, EBH-5",
        "Public contract import boundaries",
        ("intergrax/contracts/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "",
        (
            _nid(_EBH6_REG, "test_ebh6_prohibited_unresolved_zero"),
            _nid(_BND, "test_hardening_3_contracts_do_not_import_runtime_except_allowlist"),
            _nid(
                "tests/unit/architecture/test_ebh_2i_final_rescan_gate.py",
                "test_ebh_2i_contract_surface_purity_gate_passes",
            ),
        ),
        "P1",
        "Register PROHIBITED=0 + tier import boundary + EBH-2I contract surface purity",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "TYP",
        "Strong typing",
        "FRZ-TYP-*",
        "EBH-2*, GOV-X2",
        "Public seam typing",
        ("intergrax/runtime/nexus/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "Path B remediated @ HEAD — E01/Protocol 03 independent acceptance pending",
        (
            _Q2D2_REMEDIATION_NID,
            _nid(
                "tests/unit/runtime/nexus/test_authoritative_decision_exposure_propagation.py",
                "test_b13_ambiguous_candidates_fail_closed",
            ),
            _nid(
                "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py",
                "test_ebh6_application_services_production_seam_typing_passes",
            ),
        ),
        "P3-if-mandatory",
        "Q2-D2 Path B FRZ-TYP-03 remediation + APP-TYP registry (E02-OP-01) — independent acceptance pending",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PASS_SCOPED,
    ),
    ArchitectureFamilyRecertRow(
        "PLG",
        "Pluginability",
        "FRZ-PLG-*",
        "EBH-2*, EBH-5",
        "Plugin SPI + composition",
        ("intergrax/integrations/", "intergrax/tools/"),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "",
        (_nid(_EBH5_MECH, "test_ebh5_exit_inventory_complete"),),
        "P1",
        "No global PASS promotion; no regression vs EBH-5",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.REPLAY_PASS,
    ),
    ArchitectureFamilyRecertRow(
        "RPL",
        "Replaceability",
        "FRZ-RPL-*",
        "EBH-5 primary",
        "VBN inventory + matrix @ HEAD",
        ("tests/qualification/ebh_5/",),
        EBH5_INDEPENDENT_AUDIT_SHA,
        FamilyEvidenceFreshness.CURRENT,
        "51 matrix rows NOT YET QUALIFIED — EBH-7 scope",
        (_nid(_EBH5_RPLG, "test_ebh5_rpl_g_config_x_active_blockers_remain_empty"),),
        "P4",
        "Hardcoding re-audit only; matrix drift documented",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.REPLAY_PASS,
    ),
    ArchitectureFamilyRecertRow(
        "EXE",
        "Execution",
        "FRZ-EXE-* / execution invariants",
        "HARNESS-*, GOV-X2",
        "Fail-closed execution identity",
        ("intergrax/runtime/execution/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "EBH6-BLK-L9-EXE-01",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "No shadow execution path sample regression",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "GOV",
        "Governance",
        "FRZ-GOV-*",
        "GOV-X2",
        "Governance ≠ Execution separation",
        ("intergrax/runtime/governance/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "EBH6-BLK-L9-GOV-01",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Subset replay — no new bypass",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "STA",
        "State",
        "FRZ-STA-*",
        "STATE-X",
        "Canonical stores",
        ("intergrax/runtime/", "intergrax/integrations/providers/"),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "",
        (_nid(_EBH6_MECH, "test_ebh6_exit_sqlite_annex_accountability"),),
        "P2",
        "SQLite touchpoints classified; no new shadow store",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PASS_SCOPED,
    ),
    ArchitectureFamilyRecertRow(
        "REC",
        "Recovery",
        "FRZ-REC-*",
        "STATE-X, DIST-SCALE-X evidence",
        "Retry/idempotency boundaries",
        ("intergrax/runtime/resilience/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "EBH6-BLK-L9-REC-01",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Drift-only replay",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "CMP",
        "Compatibility",
        "FRZ-CMP-*",
        "COMPAT-X",
        "Compatibility policy unchanged",
        ("intergrax/runtime/compatibility/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Policy unchanged — consumer impact not EVOL scope",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.NA_WITH_EVIDENCE,
    ),
    ArchitectureFamilyRecertRow(
        "CFG",
        "Configuration",
        "FRZ-CFG-*",
        "CONFIG-X + EBH-5",
        "CONFIG-X reconciliation not PASS rewrite",
        ("tests/qualification/config_x/",),
        CONFIG_X_HISTORICAL_PASS_SHA,
        FamilyEvidenceFreshness.CURRENT,
        "",
        (_nid(_EBH6_CFG, "test_ebh6_config_x_all_concerns_mapped_at_head"),),
        "P4",
        "54/54 mapped; historical PASS preserved",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.REPLAY_PASS,
    ),
    ArchitectureFamilyRecertRow(
        "SEC",
        "Security",
        "FRZ-SEC-*",
        "CTRL-X, TENANT-X",
        "Tenant/credential boundary propagation",
        ("intergrax/runtime/security/",),
        "318a4d835534ef5c48575f4649551c957dacd8ea"[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "EBH6-BLK-L9-SEC-01",
        (
            _nid(
                "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py",
                "test_r1_sec_01_canonical_composition_rejects_external_fail_open",
            ),
        ),
        "P1",
        "Spot-check only — CTRL-X CLOSED @ historical SHA",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "REL",
        "Reliability",
        "FRZ-REL-*",
        "HARNESS-W4, CTRL-X",
        "Backpressure / bounded concurrency",
        ("intergrax/runtime/resilience/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "EBH6-BLK-L9-REL-01",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Selected guard replay",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "CTL",
        "Control planes",
        "FRZ-CTL-*",
        "CTRL-X",
        "Twelve-plane drift check",
        (_CTRL_CATALOG,),
        "318a4d835534ef5c48575f4649551c957dacd8ea",
        FamilyEvidenceFreshness.HISTORICAL,
        "",
        (
            _nid(
                "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py",
                "test_r2_mw_05_fail_open_rejected_before_attach",
            ),
        ),
        "P1",
        "Drift check only",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.REPLAY_PASS,
    ),
    ArchitectureFamilyRecertRow(
        "OBS",
        "Observability",
        "FRZ-OBS-*",
        "HARNESS-W5, TRACE-X, CTRL-X",
        "Non-authoritative export; class G",
        ("intergrax/runtime/observability/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "",
        (_nid(_EBH5_MECH, "test_ebh5_exit_qualified_rows_have_behavioral_refs"),),
        "P1",
        "OTLP qualified row replay @ HEAD",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "TRC",
        "Traceability",
        "FRZ-TRC-*",
        "TRACE-X, GOV-X2",
        "Correlation / evidence spine",
        ("tests/qualification/trace_x/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "",
        (_nid(_EBH6_MECH, "test_ebh6_exit_sanctioned_composition_roots_non_empty"),),
        "P1",
        "Sanctioned roots wired for L3",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PASS,
    ),
    ArchitectureFamilyRecertRow(
        "PRD",
        "Production qual (arch)",
        "FRZ-PRD-*",
        "PROD-Q scoped",
        "Architecture bypass census",
        ("applications/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "Live provider slice OUT OF SCOPE",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P6",
        "Live slice excluded from EBH-6 closure",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.NA_WITH_EVIDENCE,
    ),
    ArchitectureFamilyRecertRow(
        "REG",
        "Regression infra",
        "FRZ-REG-*",
        "QUAL-X OPEN",
        "Family-specific regression hooks only",
        ("tests/qualification/ebh_6/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "QUAL-X not closed by EBH-6",
        (_nid(_EBH6_MECH, "test_ebh6_exit_mandatory_docs_present"),),
        "P5",
        "EBH-6 gates only — not QUAL-X closure",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING_TO_REPLAY_PASS,
    ),
    ArchitectureFamilyRecertRow(
        "HRN",
        "Harness",
        "FRZ-HRN-*",
        "HARNESS-*",
        "W5/W6 scoped contributions",
        ("tests/qualification/harness_02/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "EBH6-BLK-L9-HRN-01",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Subset replay",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
    ArchitectureFamilyRecertRow(
        "TEN",
        "Tenant",
        "FRZ-TEN-*",
        "TENANT-X",
        "Cross-cutting propagation on audited paths",
        ("tests/unit/runtime/security/test_tenant_scope.py",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.HISTORICAL,
        "EBH6-BLK-L9-TEN-01",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Spot-check — no global TEN promotion",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.REPLAY_PASS_SCOPED,
    ),
    ArchitectureFamilyRecertRow(
        "DOC",
        "Documentation",
        "FRZ-DOC-*",
        "EBH-6, ARCH-FREEZE",
        "FRZ-DOC-01..06 current-HEAD gap analysis",
        ("docs/project/maintainers/qualification/",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "",
        (_nid(_EBH6_MECH, "test_ebh6_exit_mandatory_docs_present"),),
        "P6",
        "Dossier + matrix docs present",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PASS,
    ),
    ArchitectureFamilyRecertRow(
        "VBN",
        "Vendor neutrality rollup",
        "(VBN)",
        "EBH-6",
        "CORE_HARDCODING + CURRENT_HEAD audit",
        ("tests/qualification/ebh_6/_findings_register.py",),
        EBH5_INDEPENDENT_AUDIT_SHA[:12] + "…",
        FamilyEvidenceFreshness.CURRENT,
        "",
        (
            _nid(_EBH6_REG, "test_ebh6_findings_unclassified_zero"),
            _nid(_EBH6_REG, "test_ebh6_prohibited_unresolved_zero"),
        ),
        "P2-P4",
        "unclassified=0; prohibited unresolved=0 in frozen scope",
        FamilyIndependentDisposition.PENDING,
        ImplementerProposedDisposition.PENDING,
    ),
)


def _merge_catalog_with_l9_bundles(
    base: tuple[ArchitectureFamilyRecertRow, ...],
) -> tuple[ArchitectureFamilyRecertRow, ...]:
    from tests.qualification.ebh_6._family_l9_evidence import family_l9_bundle_by_id

    bundles = family_l9_bundle_by_id()
    merged: list[ArchitectureFamilyRecertRow] = []
    for row in base:
        bundle = bundles[row.family_id]
        gates = bundle.gate_test_ids if bundle.gate_test_ids else row.gate_test_ids
        missing = row.missing_proof
        if bundle.blocker_id and bundle.gate_test_ids:
            missing = f"{bundle.blocker_id} — invariant gates @ HEAD (implementer L9 annex)"
        elif bundle.gate_test_ids and missing.startswith("EBH6-BLK"):
            missing = ""
        merged.append(
            ArchitectureFamilyRecertRow(
                row.family_id,
                row.ecap_theme,
                row.frz_family,
                row.primary_historical_owner,
                row.authority,
                row.scope_paths,
                row.existing_evidence_sha,
                row.freshness,
                missing,
                gates,
                row.remediation_batch,
                row.acceptance_rule,
                row.independent_disposition,
                bundle.implementer_proposed_disposition,
                row.notes,
            ),
        )
    return tuple(merged)


EBH6_FAMILY_CATALOG: Final[tuple[ArchitectureFamilyRecertRow, ...]] = _merge_catalog_with_l9_bundles(
    _EBH6_FAMILY_CATALOG_BASE,
)


def family_by_id() -> dict[str, ArchitectureFamilyRecertRow]:
    return {row.family_id: row for row in EBH6_FAMILY_CATALOG}


EBH6_APPLICABLE_FAMILY_COUNT: Final[int] = len(EBH6_FAMILY_CATALOG)
