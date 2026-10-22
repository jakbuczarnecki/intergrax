# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P5 — enterprise freeze qualification suite assembly (Q12, FRZ-REG-10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.pytest_node_integrity import (
    PytestExecuteSummary,
    parse_collect_only_q,
    pytest_nodes_missing_from_collection,
    run_pytest_collect_only_q,
)
from tests.qualification.qual_x._p5_freeze_selection_catalog import (
    QUAL_X_P5_CLEAN_CHECKOUT_PROCEDURE,
    QUAL_X_P5_FREEZE_SELECTION_MANIFEST,
    QUAL_X_P5_FRZ_REG_EVIDENCE_LEDGER,
    QUAL_X_P5_START_HEAD,
    QualXFreezeMandatoryStatus,
    QualXFreezeQualificationDimension,
    QualXFreezeSelectionEntry,
    qual_x_p5_all_p5_integrity_pytest_node_ids,
    qual_x_p5_dimension_a_pytest_module_targets,
    qual_x_p5_mandatory_collectable_node_ids,
)
from tests.qualification.qual_x._p5_freeze_selection_support import (
    QualXP4Outcome,
    covered_qual_x_responsibilities,
    dimension_a_node_ids_excluding_dimension_b,
    mandatory_outcome_from_pytest_execute_summary,
    mandatory_qualification_allows_pass,
    missing_modules_for_manifest,
    qualification_sha_matches_baseline,
    reconcile_manifest_integrity,
    validate_frz_reg_ledger_complete,
    validate_manifest_dependencies,
    validate_manifest_frz_criteria,
    validate_manifest_owning_stages,
    validate_manifest_selection_ids,
)
from tests.qualification.qual_x._p3_inventory_integrity_support import (
    reconcile_harness_qinf_runtime_tool_invoker_callsites,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPECTED_Q = tuple(f"Q{index:02d}" for index in range(1, 13))


def test_qual_x_p5_start_head_documented() -> None:
    assert QUAL_X_P5_START_HEAD == "8910f580d8c53a7eb6aac0c61029f9a752c26f3a"


def test_qual_x_p5_manifest_selection_ids_unique() -> None:
    assert validate_manifest_selection_ids(QUAL_X_P5_FREEZE_SELECTION_MANIFEST) == ()
    assert reconcile_manifest_integrity() == ()


def test_qual_x_p5_covers_q01_through_q12() -> None:
    covered = covered_qual_x_responsibilities(QUAL_X_P5_FREEZE_SELECTION_MANIFEST)
    assert covered == frozenset(_EXPECTED_Q)


def test_qual_x_p5_frz_reg_ledger_maps_all_reg_criteria() -> None:
    assert validate_frz_reg_ledger_complete() == ()
    manifest_ids = {entry.selection_id for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST}
    for row in QUAL_X_P5_FRZ_REG_EVIDENCE_LEDGER:
        for selection_id in row.owning_selection_ids:
            assert selection_id in manifest_ids


def test_qual_x_p5_manifest_entries_have_frz_criteria() -> None:
    assert validate_manifest_frz_criteria(QUAL_X_P5_FREEZE_SELECTION_MANIFEST) == ()


def test_qual_x_p5_known_owning_stages_only() -> None:
    assert validate_manifest_owning_stages(QUAL_X_P5_FREEZE_SELECTION_MANIFEST) == ()


def test_qual_x_p5_dependency_chain_resolvable() -> None:
    assert validate_manifest_dependencies(QUAL_X_P5_FREEZE_SELECTION_MANIFEST) == ()


def test_qual_x_p5_dimension_b_excluded_from_dimension_a_nodes() -> None:
    dim_a_nodes = dimension_a_node_ids_excluding_dimension_b(
        QUAL_X_P5_FREEZE_SELECTION_MANIFEST
    )
    dim_b_entries = [
        entry
        for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST
        if entry.qualification_dimension
        is QualXFreezeQualificationDimension.LIVE_PROVIDER_DIMENSION_B
    ]
    assert dim_b_entries
    for entry in dim_b_entries:
        assert entry.mandatory_status is QualXFreezeMandatoryStatus.EXCLUDED_FROM_DIMENSION_A
    assert frozenset(dim_a_nodes) == frozenset(qual_x_p5_mandatory_collectable_node_ids())


def test_qual_x_p5_historical_linkage_cannot_mandatory_pass_alone() -> None:
    for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST:
        if entry.selection_id.startswith("QUAL-X-P5-DIM-B"):
            continue
        if "HISTORICAL" in entry.expected_evidence_class.value:
            assert entry.mandatory_status in (
                QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
                QualXFreezeMandatoryStatus.MANDATORY_EXECUTE_AT_QUAL_X_FINAL,
            )


def test_qual_x_p5_mandatory_collectable_nodes_exist() -> None:
    node_ids = qual_x_p5_mandatory_collectable_node_ids()
    assert node_ids
    missing, proc = pytest_nodes_missing_from_collection(node_ids, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, sorted(missing)


def test_qual_x_p5_p5_integrity_nodes_are_collectable() -> None:
    node_ids = qual_x_p5_all_p5_integrity_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(node_ids, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, sorted(missing)


def test_qual_x_p5_synthetic_duplicate_selection_id_would_fail() -> None:
    dup = QUAL_X_P5_FREEZE_SELECTION_MANIFEST[0]
    errors = validate_manifest_selection_ids((dup, dup))
    assert errors == ("duplicate selection_id: " + dup.selection_id,)


def test_qual_x_p5_synthetic_missing_frz_mapping_would_fail() -> None:
    base = QUAL_X_P5_FREEZE_SELECTION_MANIFEST[0]
    broken = QualXFreezeSelectionEntry(
        selection_id="SYNTH-NO-FRZ",
        freeze_family=base.freeze_family,
        qual_x_responsibilities=base.qual_x_responsibilities,
        owning_stage=base.owning_stage,
        invariant_summary=base.invariant_summary,
        frz_criteria=(),
        qualification_dimension=base.qualification_dimension,
        mandatory_status=base.mandatory_status,
        environment_prerequisites=base.environment_prerequisites,
        expected_evidence_class=base.expected_evidence_class,
        evidence_status=base.evidence_status,
        acceptance_boundary=base.acceptance_boundary,
        depends_on=base.depends_on,
        pytest_modules=base.pytest_modules,
        delegated_node_registry=base.delegated_node_registry,
        coordination_batch_module=base.coordination_batch_module,
    )
    assert validate_manifest_frz_criteria((broken,)) == (
        "missing frz_criteria: SYNTH-NO-FRZ",
    )


def test_qual_x_p5_synthetic_unknown_owner_would_fail() -> None:
    base = QUAL_X_P5_FREEZE_SELECTION_MANIFEST[0]
    broken = QualXFreezeSelectionEntry(
        selection_id="SYNTH-UNKNOWN-OWNER",
        freeze_family=base.freeze_family,
        qual_x_responsibilities=base.qual_x_responsibilities,
        owning_stage="UNKNOWN-STAGE-XYZ",
        invariant_summary=base.invariant_summary,
        frz_criteria=base.frz_criteria,
        qualification_dimension=base.qualification_dimension,
        mandatory_status=base.mandatory_status,
        environment_prerequisites=base.environment_prerequisites,
        expected_evidence_class=base.expected_evidence_class,
        evidence_status=base.evidence_status,
        acceptance_boundary=base.acceptance_boundary,
        depends_on=base.depends_on,
        pytest_modules=base.pytest_modules,
        delegated_node_registry=base.delegated_node_registry,
        coordination_batch_module=base.coordination_batch_module,
    )
    errors = validate_manifest_owning_stages((broken,))
    assert errors and "unknown owning_stage" in errors[0]


def test_qual_x_p5_synthetic_stale_mandatory_node_would_fail_collectability() -> None:
    fake = (
        "tests/qualification/qual_x/test_qual_x_p5_freeze_qualification_suite_assembly.py"
        "::test_never_existed_p5_mandatory_node"
    )
    missing, _proc = pytest_nodes_missing_from_collection((fake,), _REPO_ROOT)
    assert fake in missing


def test_qual_x_p5_mandatory_skip_forbidden_by_outcome_contract() -> None:
    outcome = mandatory_outcome_from_pytest_execute_summary(
        PytestExecuteSummary(passed=2, skipped=1),
        expected_passed=3,
    )
    assert outcome is QualXP4Outcome.INVALID_QUALIFICATION
    assert not mandatory_qualification_allows_pass(outcome)


def test_qual_x_p5_partial_execution_not_pass() -> None:
    outcome = mandatory_outcome_from_pytest_execute_summary(
        PytestExecuteSummary(passed=1),
        expected_passed=3,
    )
    assert outcome is QualXP4Outcome.NOT_EXECUTED


def test_qual_x_p5_mismatched_sha_not_pass() -> None:
    outcome = qualification_sha_matches_baseline(
        baseline_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        observed_sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    assert outcome is QualXP4Outcome.INVALID_QUALIFICATION
    assert not mandatory_qualification_allows_pass(outcome)


def test_qual_x_p5_clean_checkout_procedure_declares_partial() -> None:
    assert "PARTIAL" in QUAL_X_P5_CLEAN_CHECKOUT_PROCEDURE.verification_status
    assert "QUAL-X-FINAL" in QUAL_X_P5_CLEAN_CHECKOUT_PROCEDURE.mandatory_qual_x_final_note


def test_qual_x_p5_new_architecture_surface_reconciled_by_harness_qinf() -> None:
    unauthorized, stale = reconcile_harness_qinf_runtime_tool_invoker_callsites()
    assert unauthorized == []
    assert stale == []


def test_qual_x_p5_dimension_a_execution_targets_modules_exist() -> None:
    missing = missing_modules_for_manifest(_REPO_ROOT)
    assert missing == ()
    targets = qual_x_p5_dimension_a_pytest_module_targets()
    assert (
        "tests/qualification/qual_x/test_qual_x_p5_freeze_qualification_suite_assembly.py"
        in targets
    )
    assert len(targets) >= 6


def test_qual_x_p5_dimension_a_targets_cover_mandatory_collectable_proof_nodes() -> None:
    """QUAL-X-FINAL: module targets must select delegated P2/P3/P4 proof nodes, not only batches."""
    mandatory = qual_x_p5_mandatory_collectable_node_ids()
    targets = qual_x_p5_dimension_a_pytest_module_targets()
    proc = run_pytest_collect_only_q(targets, _REPO_ROOT)
    collected = parse_collect_only_q(proc.stdout)
    missing = [node_id for node_id in mandatory if node_id not in collected]
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, f"missing={len(missing)}: {sorted(missing)[:8]}"
