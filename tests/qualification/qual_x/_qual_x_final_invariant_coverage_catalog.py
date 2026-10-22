# © Artur Czarnecki. All rights reserved.

"""QUAL-X-FINAL — closed-world invariant ↔ proof disposition (P3/P4/P5 owners only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

QUAL_X_FINAL_BATCH_START_HEAD: str = "d9da2d91972788842c82b799932e933f0f5011e7"
QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA: str = (
    "9a175177419a396398647e356a08c966f9c2af4e"
)
QUAL_X_FINAL_W2_C_REMEDIATION_SHA: str = "75b8f264eec3af4e608d0618a9dfaf02bf1659bc"


class QualXFinalInvariantDisposition(StrEnum):
    CURRENT_EXECUTABLE_PROOF = "CURRENT EXECUTABLE PROOF"
    VALID_REUSED_EVIDENCE = "VALID REUSED EVIDENCE"
    HISTORICAL_ONLY = "HISTORICAL ONLY"
    MISSING_PROOF = "MISSING PROOF"
    STALE_INVENTORY = "STALE INVENTORY"
    UNJUSTIFIED_ALLOWLIST = "UNJUSTIFIED ALLOWLIST"
    FUTURE_STAGE_NOT_YET_APPLICABLE = "FUTURE STAGE — NOT YET APPLICABLE"
    DIMENSION_B_LIVE_QUALIFICATION = "DIMENSION B — LIVE QUALIFICATION"


@dataclass(frozen=True, slots=True)
class QualXFinalClosedWorldRow:
    row_id: str
    inventory_owner_id: str | None
    invariant_summary: str
    stage: str
    semantic_owner: str
    contract_or_ssot: str
    production_enforcement: str
    test_gate: str
    evidence_sha: str
    frz_criterion: str
    disposition: QualXFinalInvariantDisposition


QUAL_X_FINAL_CLOSED_WORLD_INVENTORY: tuple[QualXFinalClosedWorldRow, ...] = (
    QualXFinalClosedWorldRow(
        row_id="CW-HARNESS-QINF-NEXUS",
        inventory_owner_id="HARNESS-QINF-01",
        invariant_summary="Higher-layer Nexus importer inventory is bidirectional and complete",
        stage="HARNESS-01",
        semantic_owner="HARNESS-QINF-01",
        contract_or_ssot="tests/qualification/harness_01/nexus_import_inventory.py",
        production_enforcement="Governed harness import surfaces (inventory rows)",
        test_gate="tests/qualification/harness_01/test_harness_01_gates.py",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-05",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-HARNESS-QINF-RTI",
        inventory_owner_id="HARNESS-QINF-01",
        invariant_summary="RuntimeToolInvoker references and governed callsites are allowlisted minimally",
        stage="HARNESS-01",
        semantic_owner="HARNESS-QINF-01",
        contract_or_ssot="tests/qualification/harness_01/catalog.py",
        production_enforcement="intergrax/ + agents/ production invoke surfaces",
        test_gate="test_qual_x_p3_harness_qinf_runtime_tool_invoker_*",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-04",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-EBH-2I-DEBT",
        inventory_owner_id="EBH-2I",
        invariant_summary="Public contract dependency debt registry matches live violations",
        stage="EBH-2I",
        semantic_owner="EBH-2I",
        contract_or_ssot="testing_support/architecture/public_contract_boundary/debt_registry.py",
        production_enforcement="intergrax/contracts/** import graph",
        test_gate="tests/unit/architecture/test_ebh_2i_final_rescan_gate.py",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-04",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-HARDENING-3-LAYER",
        inventory_owner_id="HARDENING-3",
        invariant_summary="contracts/** must not import intergrax.runtime except documented allowlist",
        stage="HARDENING-3",
        semantic_owner="Runtime architecture / QUAL-X-P3",
        contract_or_ssot="tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py",
        production_enforcement="intergrax/contracts/** layer boundary",
        test_gate="test_hardening_3_contracts_do_not_import_runtime_except_allowlist",
        evidence_sha=QUAL_X_FINAL_BATCH_START_HEAD,
        frz_criterion="FRZ-REG-04",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-STATE-X-FAMILY",
        inventory_owner_id="STATE-X-INVENTORY",
        invariant_summary="STATE-X family inventory covers mandatory families without stale ids",
        stage="STATE-X",
        semantic_owner="STATE-X-INVENTORY",
        contract_or_ssot="tests/qualification/state_x/inventory.py",
        production_enforcement="STATE-X state-family SSOT",
        test_gate="test_qual_x_p3_state_x_family_inventory_mandatory_coverage",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-05",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-P2-INGRESS-LINK",
        inventory_owner_id="QUAL-X-P2-CATALOG",
        invariant_summary="QUAL-X ingress proof catalog modules exist without duplicate node ids",
        stage="QUAL-X-P2",
        semantic_owner="QUAL-X-P2-CATALOG",
        contract_or_ssot="tests/qualification/qual_x/_p2_ingress_catalog.py",
        production_enforcement="Governance/tenant ingress proof linkage",
        test_gate="test_qual_x_p3_p2_catalog_proof_modules_exist_and_no_duplicate_node_ids",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-02",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-W2-C-STORM-HYGIENE",
        inventory_owner_id=None,
        invariant_summary="W2-C retry storm test has no unhandled thread exceptions after R3",
        stage="QUAL-X-P4",
        semantic_owner="QUAL_X_P4_W2_C_DETERMINISM_DEBT",
        contract_or_ssot="tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py",
        production_enforcement="intergrax/llm_adapters/_shared/resilience.py (unchanged semantics)",
        test_gate="test_retry_storm_caps_provider_calls",
        evidence_sha=QUAL_X_FINAL_W2_C_REMEDIATION_SHA,
        frz_criterion="FRZ-REG-06",
        disposition=QualXFinalInvariantDisposition.VALID_REUSED_EVIDENCE,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-W2-C-P4-VISIBILITY",
        inventory_owner_id=None,
        invariant_summary="Dimension A exposes W2-C disposition without equating COLLECTABLE with storm execution",
        stage="QUAL-X-P4",
        semantic_owner="QUAL-X-P4 catalog",
        contract_or_ssot="tests/qualification/qual_x/_p4_determinism_environment_catalog.py",
        production_enforcement="Qualification metadata only",
        test_gate="test_qual_x_p4_w2_c_disposition_r3_remediated_test_hygiene",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-06",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-P5-MANIFEST",
        inventory_owner_id=None,
        invariant_summary="Mandatory freeze qualification suite selection is SSOT for Dimension A modules",
        stage="QUAL-X-P5",
        semantic_owner="QUAL-X",
        contract_or_ssot="tests/qualification/qual_x/_p5_freeze_selection_catalog.py",
        production_enforcement="Qualification manifest only",
        test_gate="tests/qualification/qual_x/test_qual_x_p5_freeze_qualification_suite_assembly.py",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-10",
        disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-DIM-A-FULL-BATCH",
        inventory_owner_id=None,
        invariant_summary="Full Dimension A mandatory suite from clean checkout",
        stage="QUAL-X-FINAL",
        semantic_owner="QUAL_X_P5_CLEAN_CHECKOUT_PROCEDURE",
        contract_or_ssot="P5 dimension_a_execution_command",
        production_enforcement="N/A — qualification execution",
        test_gate="535 delegated nodes @ EXECUTION_SHA",
        evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        frz_criterion="FRZ-REG-07",
        disposition=QualXFinalInvariantDisposition.VALID_REUSED_EVIDENCE,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-PROD-Q-LIVE-HTTP",
        inventory_owner_id=None,
        invariant_summary="Live GCA HTTP / provider integrated qualification",
        stage="PROD-Q",
        semantic_owner="PROD-Q-R1-R7-Q",
        contract_or_ssot="tests/qualification/prod_q/* (Dimension B selections)",
        production_enforcement="applications GCA HTTP surfaces",
        test_gate="test_prod_q_r1_r7_q_c_dim8_cross_tenant_integrated_gate.py",
        evidence_sha="—",
        frz_criterion="FRZ-REG-08",
        disposition=QualXFinalInvariantDisposition.DIMENSION_B_LIVE_QUALIFICATION,
    ),
    QualXFinalClosedWorldRow(
        row_id="CW-PLATFORM-FRZ-ROLLUP",
        inventory_owner_id=None,
        invariant_summary="All freeze checklist invariants outside P3/P5 closed-world",
        stage="ENT-AUDIT-X (future)",
        semantic_owner="QUAL-X + roadmap",
        contract_or_ssot="PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md",
        production_enforcement="Platform-wide",
        test_gate="Not in P3/P5 closed-world scan boundary",
        evidence_sha="—",
        frz_criterion="FRZ-REG-01",
        disposition=QualXFinalInvariantDisposition.MISSING_PROOF,
    ),
)


def qual_x_final_missing_p3_inventory_owner_ids() -> tuple[str, ...]:
    from tests.qualification.qual_x._p3_inventory_integrity_catalog import (
        QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS,
    )

    required = {owner.owner_id for owner in QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS}
    covered = {
        row.inventory_owner_id
        for row in QUAL_X_FINAL_CLOSED_WORLD_INVENTORY
        if row.inventory_owner_id is not None
    }
    return tuple(sorted(required - covered))
