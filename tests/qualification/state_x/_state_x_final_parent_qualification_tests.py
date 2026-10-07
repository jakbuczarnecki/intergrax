# © Artur Czarnecki. All rights reserved.

"""STATE-X-FINAL parent gates (SXF-Q01..Q35)."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

from tests.qualification.state_x._r6_recovery_branching_support import (
    STATE_X_R6_PRE_AUDIT_HEAD,
    assert_frz_rec_05_r6_completeness,
)
from tests.qualification.state_x._state_x_final_support import (
    ACCEPTED_CHILD_CHAIN,
    ATOMICITY_MATRIX,
    CONFIGURED_EFFECTIVE_PERSISTED_MATRIX,
    FAMILY_OWNERSHIP_MATRIX,
    HISTORICAL_DEBT_RECONCILIATION,
    POLICY_DURABILITY_MATRIX,
    PRIMARY_FRZ_CRITERION_EVIDENCE,
    R1_SQLITE_ENV_01,
    R1SqliteEnv01Disposition,
    TENANT_ISOLATION_AUDIT_FINAL,
    assert_family_inventory_closed_world,
    assert_no_duplicate_semantic_owners,
    assert_no_in_scope_state_x_blockers,
    assert_primary_frz_matrix_complete,
    assert_r1_sqlite_disposition_final,
    assert_state_x_final_mechanical_gate,
    scan_unclassified_durable_persistence_paths,
)
from tests.qualification.state_x.inventory import (
    CURRENT_STATE_X_FAMILY_IDS,
    HISTORICAL_BASE_FAMILY_IDS,
    MANDATORY_FAMILY_IDS,
    STATE_X_FINAL_ALLOWLIST_PATHS,
    STATE_X_FINAL_R1_R1_ALLOWLIST_PATHS,
    STATE_X_FINAL_START_HEAD,
    STATE_X_R6_ACCEPTED_CLOSURE_SHA,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_test(module_path: str, test_name: str) -> None:
    mod = importlib.import_module(module_path)
    fn = getattr(mod, test_name, None)
    assert callable(fn), f"missing evidence {module_path}.{test_name}"


def test_sxf_q01_r6_closure_reconciled() -> None:
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            STATE_X_R6_ACCEPTED_CLOSURE_SHA,
            STATE_X_FINAL_START_HEAD,
        ],
        cwd=_REPO_ROOT,
        check=False,
    )
    assert ancestor.returncode == 0, (
        f"R6 closure {STATE_X_R6_ACCEPTED_CLOSURE_SHA} must be ancestor of "
        f"STATE-X-FINAL start {STATE_X_FINAL_START_HEAD}"
    )
    assert STATE_X_R6_PRE_AUDIT_HEAD == "bcd8157065cc649412b64e9d6ada34be92d4b6a3"
    assert_frz_rec_05_r6_completeness()


def test_sxf_q02_sx_families_complete() -> None:
    assert_family_inventory_closed_world()
    assert set(HISTORICAL_BASE_FAMILY_IDS) <= set(CURRENT_STATE_X_FAMILY_IDS)
    assert {r.family_id for r in FAMILY_OWNERSHIP_MATRIX} == set(CURRENT_STATE_X_FAMILY_IDS)


def test_sxf_q03_unclassified_durable_paths_zero() -> None:
    assert scan_unclassified_durable_persistence_paths() == ()


def test_sxf_q04_exactly_one_semantic_owner() -> None:
    assert_no_duplicate_semantic_owners()
    assert len(FAMILY_OWNERSHIP_MATRIX) == len(CURRENT_STATE_X_FAMILY_IDS)


def test_sxf_q05_duplicate_semantic_store_zero() -> None:
    assert all(not r.duplicate_authority for r in FAMILY_OWNERSHIP_MATRIX)


def test_sxf_q06_atomicity_matrix_complete() -> None:
    assert len(ATOMICITY_MATRIX) >= 5
    for row in ATOMICITY_MATRIX:
        assert row.atomic_write_unit
        assert "distributed" not in row.cross_store_atomicity.lower() or "no" in row.cross_store_atomicity.lower()


def test_sxf_q07_r1_sqlite_env_01_classified() -> None:
    assert_r1_sqlite_disposition_final()
    assert (
        R1_SQLITE_ENV_01.disposition
        is R1SqliteEnv01Disposition.CLASSIFIED_ENVIRONMENT_NO_STATE_X_IMPACT
    )


def test_sxf_q08_tenant_isolation_pass() -> None:
    assert TENANT_ISOLATION_AUDIT_FINAL.result == "PASS"
    assert TENANT_ISOLATION_AUDIT_FINAL.cross_tenant_path == "DENIED"


def test_sxf_q09_stale_state_evidence_registered() -> None:
    _import_test(
        "tests.qualification.state_x._r3_r2_qualification_tests",
        "test_r3_r2_q17_compensation_fence_supersession",
    )


def test_sxf_q10_configured_effective_persisted_complete() -> None:
    assert len(CONFIGURED_EFFECTIVE_PERSISTED_MATRIX) >= 4


def test_sxf_q11_policy_durability_complete() -> None:
    assert all(row.classification for row in POLICY_DURABILITY_MATRIX)
    assert all(not row.widens_authority_if_lost or row.owner for row in POLICY_DURABILITY_MATRIX)


def test_sxf_q12_checkpoint_not_identity_authority() -> None:
    _import_test(
        "tests.qualification.state_x._r6_recovery_branching_qualification_tests",
        "test_r6_q16_new_execution_mints_independent_identity",
    )


def test_sxf_q17_fork_semantics_pass() -> None:
    row = next(r for r in PRIMARY_FRZ_CRITERION_EVIDENCE if r.criterion_id == "FRZ-REC-05")
    assert row.status == "PASS"


def test_sxf_q20_backup_restore_pass() -> None:
    row = next(r for r in PRIMARY_FRZ_CRITERION_EVIDENCE if r.criterion_id == "FRZ-REC-08")
    assert row.status == "PASS"


def test_sxf_q29_eighteen_primary_frz_rows() -> None:
    assert_primary_frz_matrix_complete()
    assert len(PRIMARY_FRZ_CRITERION_EVIDENCE) == 18


def test_sxf_q30_child_chain_documented() -> None:
    assert len(ACCEPTED_CHILD_CHAIN) >= 7


def test_sxf_q31_production_delta_qualification_only() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", STATE_X_FINAL_START_HEAD],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    scope_allowlist = STATE_X_FINAL_ALLOWLIST_PATHS | STATE_X_FINAL_R1_R1_ALLOWLIST_PATHS
    for path in changed:
        norm = path.replace("\\", "/")
        if norm in scope_allowlist:
            continue
        assert norm.startswith(
            (
                "docs/project/maintainers/",
                "tests/qualification/state_x/",
            ),
        ), norm


def test_sxf_q33_in_scope_blocker_zero() -> None:
    assert_no_in_scope_state_x_blockers()


def test_sxf_q34_unclassified_zero() -> None:
    assert scan_unclassified_durable_persistence_paths() == ()


def test_sxf_q35_mechanical_parent_gate() -> None:
    assert_state_x_final_mechanical_gate()


def test_sxf_historical_debt_reconciled() -> None:
    ids = {row.debt_id for row in HISTORICAL_DEBT_RECONCILIATION}
    assert {"Q2-D1", "R1-SQLITE-ENV-01", "CTRL-X-R3-R2 state/recovery debt"} <= ids


@pytest.mark.parametrize("row", PRIMARY_FRZ_CRITERION_EVIDENCE)
def test_sxf_primary_frz_evidence_tests_exist(row) -> None:
    for test_name in row.exact_tests:
        if test_name.startswith("test_sxf_"):
            continue
        for pkg in (
            "tests.qualification.state_x._r3_r2_qualification_tests",
            "tests.qualification.state_x._r3_r3_qualification_tests",
            "tests.qualification.state_x._r3_r4_qualification_tests",
            "tests.qualification.state_x._r4_task_checkpoint_restore_qualification_tests",
            "tests.qualification.state_x._r4_r1_restore_consumer_convergence_tests",
            "tests.qualification.state_x._r5_backup_restore_qualification_tests",
            "tests.qualification.state_x._r5_q1_cross_store_restore_tests",
            "tests.qualification.state_x._r6_recovery_branching_qualification_tests",
            "tests.qualification.state_x._state_x_final_parent_qualification_tests",
        ):
            mod = importlib.import_module(pkg)
            if hasattr(mod, test_name):
                return
        raise AssertionError(f"evidence test not found: {test_name}")
