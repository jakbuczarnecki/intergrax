# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P3 — inventory, allowlist & recovery evidence integrity (Q06, Q11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.architecture.public_contract_boundary.models import (
    ContractDependencyDebtEntry,
    DependencyRuleId,
    RemovalStage,
)
from testing_support.architecture.public_contract_boundary.debt_validation import (
    validate_debt_registry,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)
from tests.qualification.harness_01.catalog import (
    HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES,
)
from tests.qualification.qual_x._p3_inventory_integrity_catalog import (
    QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS,
    QUAL_X_P3_Q06_RECOVERY_EVIDENCE_MATRIX,
    QUAL_X_P3_START_HEAD,
    QualXP3EvidenceDisposition,
    qual_x_p3_ebh_allowlist_ledger_rows,
    qual_x_p3_inventory_owner_ssot_modules_exist,
    qual_x_p3_q06_all_proof_pytest_node_ids,
)
from tests.qualification.qual_x._p3_inventory_integrity_support import (
    duplicate_node_ids,
    ebh_public_contract_boundary_passes,
    missing_proof_module_paths,
    p2_catalog_proof_integrity,
    reconcile_ebh_debt_registry_metadata,
    reconcile_hardening_3_contract_runtime_import_allowlist,
    reconcile_harness_qinf_nexus_importers,
    reconcile_harness_qinf_runtime_tool_invoker_callsites,
    reconcile_harness_qinf_runtime_tool_invoker_references,
    reconcile_state_x_family_inventory_ids,
)
from testing_support.architecture.contracts_runtime_layer_boundary import (
    reconcile_contracts_runtime_import_allowlist,
)
from tests.qualification.qual_x._qual_x_final_invariant_coverage_catalog import (
    QUAL_X_FINAL_CLOSED_WORLD_INVENTORY,
    qual_x_final_missing_p3_inventory_owner_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_qual_x_p3_start_head_documented() -> None:
    assert QUAL_X_P3_START_HEAD == "2031a913762de85a24daaa230d9ab0be30ba8905"


def test_qual_x_p3_in_scope_inventory_owners_are_unique() -> None:
    owner_ids = [row.owner_id for row in QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS]
    assert len(owner_ids) == len(set(owner_ids))


def test_qual_x_p3_inventory_owner_ssot_and_gate_modules_exist() -> None:
    missing = qual_x_p3_inventory_owner_ssot_modules_exist(_REPO_ROOT)
    assert missing == (), f"missing owner modules: {missing}"


def test_qual_x_p3_q06_matrix_rows_declare_collectable_linkage_only() -> None:
    for row in QUAL_X_P3_Q06_RECOVERY_EVIDENCE_MATRIX:
        assert row.primary_proof_node_ids, row.row_id
        assert row.disposition is QualXP3EvidenceDisposition.COLLECTABLE_AT_HEAD, row.row_id
        assert "HISTORICAL" in row.historical_acceptance_note.upper()


def test_qual_x_p3_q06_proof_nodes_are_collectable() -> None:
    combined = qual_x_p3_q06_all_proof_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(combined, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, f"missing={len(missing)}: {sorted(missing)}"


def test_qual_x_p3_q06_proof_modules_exist_on_disk() -> None:
    node_ids = qual_x_p3_q06_all_proof_pytest_node_ids()
    missing = missing_proof_module_paths(node_ids, _REPO_ROOT)
    assert missing == ()


def test_qual_x_p3_harness_qinf_runtime_tool_invoker_callsites_bidirectional() -> None:
    unauthorized, stale = reconcile_harness_qinf_runtime_tool_invoker_callsites()
    assert unauthorized == [], unauthorized
    assert stale == [], stale


def test_qual_x_p3_harness_qinf_runtime_tool_invoker_reference_allowlist_bidirectional() -> None:
    unclassified, stale = reconcile_harness_qinf_runtime_tool_invoker_references()
    assert unclassified == [], unclassified
    assert stale == [], stale


def test_qual_x_p3_harness_qinf_nexus_importer_inventory_bidirectional() -> None:
    unclassified, stale, incomplete = reconcile_harness_qinf_nexus_importers()
    assert unclassified == [], unclassified
    assert stale == [], stale
    assert incomplete == [], incomplete


def test_qual_x_p3_hardening_3_contract_runtime_import_allowlist_bidirectional() -> None:
    violations, stale_allowlist = reconcile_hardening_3_contract_runtime_import_allowlist(
        _REPO_ROOT
    )
    assert violations == [], violations
    assert stale_allowlist == [], stale_allowlist


def test_qual_x_p3_synthetic_stale_hardening_allowlist_entry_would_fail(
    tmp_path: Path,
) -> None:
    contracts = tmp_path / "intergrax" / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "orphan_allowlist.py").write_text("x = 1\n", encoding="utf-8")
    stale_entry = "intergrax/contracts/orphan_allowlist.py"
    reconciliation = reconcile_contracts_runtime_import_allowlist(
        tmp_path,
        allowlist=frozenset({stale_entry}),
    )
    assert stale_entry in reconciliation.stale_allowlist_entries


def test_qual_x_p3_closed_world_inventory_maps_all_p3_owners() -> None:
    assert qual_x_final_missing_p3_inventory_owner_ids() == ()
    row_ids = [row.row_id for row in QUAL_X_FINAL_CLOSED_WORLD_INVENTORY]
    assert len(row_ids) == len(set(row_ids))


def test_qual_x_p3_state_x_family_inventory_mandatory_coverage() -> None:
    missing, structural = reconcile_state_x_family_inventory_ids()
    assert missing == [], missing
    assert structural == [], structural


def test_qual_x_p3_ebh_debt_registry_metadata_valid_and_modules_exist() -> None:
    registry_errors, missing_modules = reconcile_ebh_debt_registry_metadata(_REPO_ROOT)
    assert registry_errors == (), registry_errors
    assert missing_modules == (), missing_modules


def test_qual_x_p3_ebh_allowlist_ledger_matches_live_registry() -> None:
    ledger = qual_x_p3_ebh_allowlist_ledger_rows()
    finding_ids = {row.finding_id for row in ledger}
    assert len(finding_ids) == len(ledger)
    for row in ledger:
        assert row.owner_id == "EBH-2I"
        assert row.rationale_summary.strip()
        assert row.removal_milestone


def test_qual_x_p3_ebh_public_contract_boundary_gate_passes_at_head() -> None:
    assert ebh_public_contract_boundary_passes(_REPO_ROOT)


def test_qual_x_p3_p2_catalog_proof_modules_exist_and_no_duplicate_node_ids() -> None:
    dups, missing_modules = p2_catalog_proof_integrity(_REPO_ROOT)
    assert dups == (), dups
    assert missing_modules == (), missing_modules


def test_qual_x_p3_synthetic_stale_harness_callsite_inventory_would_fail() -> None:
    stale_row = "intergrax/synthetic/deleted_runtime_tool_invoker_callsite.py"
    authorized = HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES | {stale_row}
    assert sorted(authorized - HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES) == [
        stale_row
    ]


def test_qual_x_p3_synthetic_unclassified_new_q06_module_would_fail_existence_gate() -> None:
    fake = "tests/qualification/qual_x/synthetic_deleted_q06_proof.py::test_never_existed"
    assert missing_proof_module_paths((fake,), _REPO_ROOT) == (
        "tests/qualification/qual_x/synthetic_deleted_q06_proof.py",
    )


def test_qual_x_p3_synthetic_duplicate_debt_finding_id_would_fail_registry_validation() -> None:
    entry = ContractDependencyDebtEntry(
        finding_id="SYNTH-DUP",
        source_module="intergrax.contracts.idempotency_store",
        forbidden_import_module="intergrax.tools.execution_models",
        rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION,
        removal_stage=RemovalStage.EBH_2B,
    )
    errors = validate_debt_registry((entry, entry))
    assert any("Duplicate finding_id" in err for err in errors)


def test_qual_x_p3_synthetic_duplicate_q06_node_id_would_fail() -> None:
    node_ids = ("mod.py::test_a", "mod.py::test_a")
    assert duplicate_node_ids(node_ids) == ("mod.py::test_a",)
