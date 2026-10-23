# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P3 — inventory / allowlist / recovery evidence linkage (Q06, Q11)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

QUAL_X_P3_START_HEAD: str = "2031a913762de85a24daaa230d9ab0be30ba8905"

_R1 = "tests/qualification/state_x/test_state_x_r1_checkpoint_resume_terminal.py"
_R4 = "tests/qualification/state_x/test_state_x_r4_task_checkpoint_restore.py"
_R6 = "tests/qualification/state_x/test_state_x_r6_recovery_branching.py"
_TXP1 = "tests/qualification/trace_x/test_trace_x_p1_identity_causality.py"
_TXP4R1 = "tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py"
_P2_CATALOG = "tests/qualification/qual_x/_p2_ingress_catalog.py"


def _nid(module: str, test_name: str) -> str:
    return f"{module}::{test_name}"


class QualXP3EvidenceDisposition(StrEnum):
    COLLECTABLE_AT_HEAD = "COLLECTABLE_AT_HEAD"
    HISTORICAL_STAGE_EXECUTION = "HISTORICAL_STAGE_EXECUTION"
    SUPPORTING_STATIC = "SUPPORTING_STATIC"


@dataclass(frozen=True, slots=True)
class QualXP3InventoryOwner:
    owner_id: str
    semantic_owner: str
    ssot_module: str
    primary_gate_module: str
    reconciliation_class: str
    frz_criteria: tuple[str, ...]


QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS: tuple[QualXP3InventoryOwner, ...] = (
    QualXP3InventoryOwner(
        owner_id="HARNESS-QINF-01",
        semantic_owner="HARNESS-01 qualification inventory",
        ssot_module="tests/qualification/harness_01/nexus_import_inventory.py",
        primary_gate_module="tests/qualification/harness_01/test_harness_01_gates.py",
        reconciliation_class="bidirectional_scan_plus_explicit_allowlist",
        frz_criteria=("FRZ-REG-04", "FRZ-REG-05"),
    ),
    QualXP3InventoryOwner(
        owner_id="EBH-2I",
        semantic_owner="Public contract boundary dependency debt registry",
        ssot_module="testing_support/architecture/public_contract_boundary/debt_registry.py",
        primary_gate_module="tests/unit/architecture/test_ebh_2i_final_rescan_gate.py",
        reconciliation_class="fail_closed_rescan_plus_debt_staleness",
        frz_criteria=("FRZ-REG-04", "FRZ-REG-02", "FRZ-BND-01"),
    ),
    QualXP3InventoryOwner(
        owner_id="STATE-X-INVENTORY",
        semantic_owner="STATE-X state-family SSOT",
        ssot_module="tests/qualification/state_x/inventory.py",
        primary_gate_module="tests/qualification/state_x/test_state_x_p0_baseline.py",
        reconciliation_class="bidirectional_family_and_path_inventory",
        frz_criteria=("FRZ-STA-01", "FRZ-REC-01", "FRZ-REG-05"),
    ),
    QualXP3InventoryOwner(
        owner_id="QUAL-X-P2-CATALOG",
        semantic_owner="QUAL-X ingress proof-linkage catalog (regression only)",
        ssot_module=_P2_CATALOG,
        primary_gate_module="tests/qualification/qual_x/test_qual_x_p2_negative_ingress_matrix.py",
        reconciliation_class="collectability_plus_module_existence",
        frz_criteria=("FRZ-REG-02", "FRZ-REG-05"),
    ),
    QualXP3InventoryOwner(
        owner_id="HARDENING-3",
        semantic_owner="Contract layer must not import runtime (documented allowlist)",
        ssot_module=(
            "testing_support/architecture/contracts_runtime_layer_boundary/allowlist.py"
        ),
        primary_gate_module="tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py",
        reconciliation_class="bidirectional_contract_runtime_import_allowlist",
        frz_criteria=("FRZ-REG-04", "FRZ-REG-05", "FRZ-BND-02"),
    ),
)


@dataclass(frozen=True, slots=True)
class QualXP3Q06EvidenceRow:
    row_id: str
    invariant: str
    canonical_owner: str
    public_contract: str
    production_surface: str
    regression_gate: str
    primary_proof_node_ids: tuple[str, ...]
    disposition: QualXP3EvidenceDisposition
    historical_acceptance_note: str
    frz_criteria: tuple[str, ...]


QUAL_X_P3_Q06_RECOVERY_EVIDENCE_MATRIX: tuple[QualXP3Q06EvidenceRow, ...] = (
    QualXP3Q06EvidenceRow(
        row_id="Q06-R1",
        invariant="Checkpoint resume rejects malformed / stale / cross-tenant snapshot authority",
        canonical_owner="STATE-X-R1",
        public_contract="intergrax.runtime.long_running.checkpoint_resume_validation",
        production_surface="intergrax/runtime/long_running/checkpoint_resume_validation.py",
        regression_gate=_R1,
        primary_proof_node_ids=(
            _nid(_R1, "test_r1_q04_empty_snapshot_reject_malformed"),
            _nid(_R1, "test_r1_q11_stale_revision_reject_stale"),
            _nid(_R1, "test_r1_q07_snapshot_tenant_mismatch_reject_tenant"),
        ),
        disposition=QualXP3EvidenceDisposition.COLLECTABLE_AT_HEAD,
        historical_acceptance_note="HISTORICAL STATE-X-R1 wave; P3 linkage only — not replayed in P3 batch",
        frz_criteria=("FRZ-REC-01", "FRZ-STA-02", "FRZ-TEN-01"),
    ),
    QualXP3Q06EvidenceRow(
        row_id="Q06-R4",
        invariant="Task checkpoint restore validates snapshot before materialization",
        canonical_owner="STATE-X-R4",
        public_contract="intergrax.runtime.long_running.task_checkpoint_restore",
        production_surface="intergrax/runtime/long_running/task_checkpoint_restore.py",
        regression_gate=_R4,
        primary_proof_node_ids=(
            _nid(_R4, "test_r4_q05_empty_task_snapshot_rejected"),
            _nid(_R4, "test_r4_q08_snapshot_task_mismatch_rejected"),
        ),
        disposition=QualXP3EvidenceDisposition.COLLECTABLE_AT_HEAD,
        historical_acceptance_note="HISTORICAL STATE-X-R4; P3 collectability gate only",
        frz_criteria=("FRZ-REC-02", "FRZ-STA-03"),
    ),
    QualXP3Q06EvidenceRow(
        row_id="Q06-R6",
        invariant="Resume preserves logical run identity and lineage after recovery branching",
        canonical_owner="STATE-X-R6",
        public_contract="intergrax.runtime.long_running.recovery_branching",
        production_surface="intergrax/runtime/long_running/recovery_branching.py",
        regression_gate=_R6,
        primary_proof_node_ids=(
            _nid(_R6, "test_r6_q05_resume_preserves_logical_run_identity"),
            _nid(_R6, "test_r6_q07_resume_historical_execution_lineage_preserved"),
        ),
        disposition=QualXP3EvidenceDisposition.COLLECTABLE_AT_HEAD,
        historical_acceptance_note="HISTORICAL STATE-X-R6 closure; linkage @ current HEAD",
        frz_criteria=("FRZ-REC-03", "FRZ-STA-04"),
    ),
    QualXP3Q06EvidenceRow(
        row_id="Q06-TRC-P1",
        invariant="Causal trace facts enforce tenant equality (identity continuity)",
        canonical_owner="TRACE-X-P1",
        public_contract="intergrax.contracts.runtime_execution_ref",
        production_surface="intergrax/contracts/runtime_execution_ref.py",
        regression_gate=_TXP1,
        primary_proof_node_ids=(
            _nid(_TXP1, "test_txp1_q05_tenant_equality_enforced_on_causal_fact"),
        ),
        disposition=QualXP3EvidenceDisposition.COLLECTABLE_AT_HEAD,
        historical_acceptance_note="HISTORICAL TRACE-X-P1; not re-executed in P3 batch",
        frz_criteria=("FRZ-TRC-02", "FRZ-TEN-02"),
    ),
    QualXP3Q06EvidenceRow(
        row_id="Q06-TRC-P4",
        invariant="Active runtime event recorder binding is tenant-scoped and cleared on stale context",
        canonical_owner="TRACE-X-P4-R1",
        public_contract="intergrax.runtime.events.active_runtime_event_recorder",
        production_surface="intergrax/runtime/events/active_runtime_event_recorder.py",
        regression_gate=_TXP4R1,
        primary_proof_node_ids=(
            _nid(_TXP4R1, "test_txp4r1_q06_sequential_tenant_isolation"),
            _nid(_TXP4R1, "test_txp4r1_ten_l4_stale_tenant_cleared_without_recorder"),
        ),
        disposition=QualXP3EvidenceDisposition.COLLECTABLE_AT_HEAD,
        historical_acceptance_note="HISTORICAL TRACE-X-P4; P3 linkage only",
        frz_criteria=("FRZ-TRC-06", "FRZ-TRC-07"),
    ),
)


@dataclass(frozen=True, slots=True)
class QualXP3AllowlistLedgerRow:
    finding_id: str
    owner_id: str
    exempted_rule: str
    source_module: str
    rationale_summary: str
    removal_milestone: str


def qual_x_p3_ebh_allowlist_ledger_rows() -> tuple[QualXP3AllowlistLedgerRow, ...]:
    from testing_support.architecture.public_contract_boundary.debt_registry import (
        PUBLIC_CONTRACT_DEPENDENCY_DEBT,
    )

    rows: list[QualXP3AllowlistLedgerRow] = []
    for entry in PUBLIC_CONTRACT_DEPENDENCY_DEBT:
        rows.append(
            QualXP3AllowlistLedgerRow(
                finding_id=entry.finding_id,
                owner_id="EBH-2I",
                exempted_rule=entry.rule_id.value,
                source_module=entry.source_module,
                rationale_summary=(
                    f"Tracked EBH-2A debt; removal_stage={entry.removal_stage.value}"
                ),
                removal_milestone=entry.removal_stage.value,
            )
        )
    return tuple(rows)


def qual_x_p3_q06_all_proof_pytest_node_ids() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in QUAL_X_P3_Q06_RECOVERY_EVIDENCE_MATRIX:
        for node_id in row.primary_proof_node_ids:
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
    return tuple(ordered)


def qual_x_p3_inventory_owner_ssot_modules_exist(repo_root) -> tuple[str, ...]:
    from pathlib import Path

    root = Path(repo_root)
    missing: list[str] = []
    for owner in QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS:
        for rel in (owner.ssot_module, owner.primary_gate_module):
            if not (root / rel).is_file():
                missing.append(rel)
    return tuple(missing)
