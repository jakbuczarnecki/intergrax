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
    notes: str = ""


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


_BND = "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
_EBH6_MECH = "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py"
_EBH6_REG = "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py"
_EBH6_CFG = "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py"
_EBH6_FAM = "tests/qualification/ebh_6/test_ebh_6_architecture_family_matrix_integrity.py"
_EBH5_MECH = "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py"
_EBH5_RPLG = "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py"
_CTRL_CATALOG = "tests/qualification/control_planes/ctrl_x/catalog.py"


EBH6_FAMILY_CATALOG: Final[tuple[ArchitectureFamilyRecertRow, ...]] = (
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
        (_nid(_BND, "test_tier0_must_not_import_agents_or_applications"),),
        "P1",
        "No new PROVEN boundary violation on production path",
        FamilyIndependentDisposition.REPLAY_PASS,
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
        "",
        (_nid(_EBH6_FAM, "test_ebh6_family_catalog_covers_applicable_themes"),),
        "P1",
        "No new duplicate semantic owner",
        FamilyIndependentDisposition.PENDING,
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
        (_nid(_EBH6_REG, "test_ebh6_no_prohibited_unresolved_in_scope"),),
        "P1",
        "No contract leakage finding PROHIBITED unresolved",
        FamilyIndependentDisposition.PENDING,
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
        "Q2-D2 ratified debt disposition",
        (_nid(_EBH6_REG, "test_ebh6_tracked_debt_findings_ratified"),),
        "P3-if-mandatory",
        "Q2-D2 RATIFIED or remediated before closure",
        FamilyIndependentDisposition.PENDING,
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
        FamilyIndependentDisposition.REPLAY_PASS,
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
        FamilyIndependentDisposition.REPLAY_PASS,
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
        "",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "No shadow execution path sample regression",
        FamilyIndependentDisposition.PENDING,
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
        "",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Subset replay — no new bypass",
        FamilyIndependentDisposition.PENDING,
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
        "",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Drift-only replay",
        FamilyIndependentDisposition.PENDING,
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
        FamilyIndependentDisposition.NOT_APPLICABLE_WITH_EVIDENCE,
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
        FamilyIndependentDisposition.REPLAY_PASS,
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
        "",
        (
            _nid(
                "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py",
                "test_r1_sec_01_canonical_composition_rejects_external_fail_open",
            ),
        ),
        "P1",
        "Spot-check only — CTRL-X CLOSED @ historical SHA",
        FamilyIndependentDisposition.PENDING,
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
        "",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Selected guard replay",
        FamilyIndependentDisposition.PENDING,
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
        FamilyIndependentDisposition.REPLAY_PASS,
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
        FamilyIndependentDisposition.NOT_APPLICABLE_WITH_EVIDENCE,
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
        "",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Subset replay",
        FamilyIndependentDisposition.PENDING,
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
        "TENANT-X global not entered",
        (_nid(_EBH6_FAM, "test_ebh6_family_row_count_closed_world"),),
        "P1",
        "Spot-check — no global TEN promotion",
        FamilyIndependentDisposition.PENDING,
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
    ),
)


def family_by_id() -> dict[str, ArchitectureFamilyRecertRow]:
    return {row.family_id: row for row in EBH6_FAMILY_CATALOG}


EBH6_APPLICABLE_FAMILY_COUNT: Final[int] = len(EBH6_FAMILY_CATALOG)
