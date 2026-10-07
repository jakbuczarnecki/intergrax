# © Artur Czarnecki. All rights reserved.

"""STATE-X-FINAL-R1 closed-world durable state gates (SXF-R1-Q01..Q50)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.qualification.state_x._r5_backup_restore_support import (
    assert_state_x_current_backup_restore_completeness,
)
from tests.qualification.state_x._state_x_closed_world_durable_state_support import (
    DISCOVERY_SCAN_ROOTS,
    PRIOR_BROAD_EXCLUSION_PREFIXES,
    STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY,
    assert_current_family_registry_complete,
    assert_durable_state_discovery_fully_classified,
    assert_no_blind_directory_exclusions_in_scanner,
    discover_durable_state_candidates,
    scan_unclassified_durable_persistence_paths,
)
from tests.qualification.state_x._state_x_final_support import (
    ATOMICITY_MATRIX,
    CONFIGURED_EFFECTIVE_PERSISTED_MATRIX,
    POLICY_DURABILITY_MATRIX,
    assert_state_x_final_mechanical_gate,
)
from tests.qualification.state_x.inventory import (
    CURRENT_STATE_X_FAMILY_IDS,
    HISTORICAL_BASE_FAMILY_IDS,
    STATE_X_FINAL_R1_START_HEAD,
    STATE_X_KNOWN_BLOCKERS,
    BlockerClassification,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_sxf_r1_q01_start_head_exact() -> None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == STATE_X_FINAL_R1_START_HEAD


def test_sxf_r1_q02_historical_families_preserved() -> None:
    assert set(HISTORICAL_BASE_FAMILY_IDS) <= set(CURRENT_STATE_X_FAMILY_IDS)


def test_sxf_r1_q03_broad_prefix_exclusions_removed() -> None:
    assert_no_blind_directory_exclusions_in_scanner()
    assert PRIOR_BROAD_EXCLUSION_PREFIXES


def test_sxf_r1_q04_discovery_roots() -> None:
    assert "intergrax/contracts" in DISCOVERY_SCAN_ROOTS
    assert "intergrax/runtime" in DISCOVERY_SCAN_ROOTS
    assert "agents" in DISCOVERY_SCAN_ROOTS
    assert "applications" in DISCOVERY_SCAN_ROOTS


def test_sxf_r1_q05_q06_discovery_equals_classified() -> None:
    assert_durable_state_discovery_fully_classified()


def test_sxf_r1_q07_canonical_owners_map_to_families() -> None:
    assert_current_family_registry_complete()


def test_sxf_r1_q11_background_identity_classified() -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.symbol == "BackgroundExecutionIdentityPersistence"
    ]
    assert hits and hits[0].family_id == "SX-F16"


def test_sxf_r1_q14_continuation_classified() -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.symbol == "ExecutionContinuationStateStore"
    ]
    assert hits and hits[0].family_id == "SX-F17"


def test_sxf_r1_q18_deadline_classified() -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.symbol == "KvExecutionDeadlinePersistence"
    ]
    assert hits and hits[0].family_id == "SX-F18"


def test_sxf_r1_q23_delegated_correlation_classified() -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.symbol == "DelegatedInvocationCorrelationStore"
    ]
    assert hits and hits[0].family_id == "SX-F19"


def test_sxf_r1_q29_task_memory_disposition() -> None:
    task_mem = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.paths[0].startswith("intergrax/runtime/task_memory/")
    ]
    assert task_mem
    assert all(r.owner_stage == "APPLICATION-MEMORY" for r in task_mem)


def test_sxf_r1_q32_unclassified_zero() -> None:
    assert scan_unclassified_durable_persistence_paths() == ()


def test_sxf_r1_q34_current_family_registry() -> None:
    assert len(CURRENT_STATE_X_FAMILY_IDS) >= len(HISTORICAL_BASE_FAMILY_IDS)


def test_sxf_r1_q35_backup_restore_current_complete() -> None:
    assert_state_x_current_backup_restore_completeness()


def test_sxf_r1_q36_atomicity_matrix_current() -> None:
    matrix_ids = {row.family_id for row in ATOMICITY_MATRIX}
    for fid in ("SX-F16", "SX-F17", "SX-F18", "SX-F19", "SX-F20"):
        assert fid in matrix_ids


def test_sxf_r1_q49_in_scope_blocker_zero() -> None:
    for blocker in STATE_X_KNOWN_BLOCKERS:
        assert blocker.classification is not BlockerClassification.IN_SCOPE_BLOCKER


def test_sxf_r1_q50_discovery_inventory_counts() -> None:
    discovered = discover_durable_state_candidates()
    assert len(discovered) == len(STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY)
    assert_durable_state_discovery_fully_classified()


def test_sxf_r1_mechanical_parent_gate() -> None:
    assert_state_x_final_mechanical_gate()
