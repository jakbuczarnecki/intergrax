# © Artur Czarnecki. All rights reserved.
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from tests.unit.campaign_b._pt_t01_matrix import (
    A4_006_TRACKED,
    ACTIVE_TRACKED_PYTEST_ARGS,
    EXPECTED_NT_IDS,
    PT_T01_MATRIX,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KNOWN_STATUSES = frozenset({"ACTIVE", "HELD_OUT", "CONDITIONAL"})


def test_pt_t01_matrix_row_count_is_canonical() -> None:
    assert len(PT_T01_MATRIX) == 18


def test_pt_t01_matrix_identifier_set_matches_canonical_inventory() -> None:
    assert {row.nt_id for row in PT_T01_MATRIX} == EXPECTED_NT_IDS


def test_pt_t01_matrix_no_duplicate_nt_identifiers() -> None:
    ids = [row.nt_id for row in PT_T01_MATRIX]
    assert len(ids) == len(set(ids))


def test_pt_t01_active_rows_have_tracked_pytest_targets() -> None:
    for row in PT_T01_MATRIX:
        if row.status == "ACTIVE":
            assert row.tracked_pytest, f"{row.nt_id}: ACTIVE without pytest targets"


def test_pt_t01_held_out_rows_have_no_targets_and_reason() -> None:
    for row in PT_T01_MATRIX:
        if row.status == "HELD_OUT":
            assert not row.tracked_pytest
            assert row.held_out_reason.strip(), f"{row.nt_id}: HELD_OUT without reason"


def test_pt_t01_conditional_rows_have_reason_and_mapping() -> None:
    for row in PT_T01_MATRIX:
        if row.status == "CONDITIONAL":
            assert row.held_out_reason.strip(), f"{row.nt_id}: CONDITIONAL without reason"
            assert row.tracked_pytest, f"{row.nt_id}: CONDITIONAL without test mapping"


def test_pt_t01_disposition_counts() -> None:
    counts = Counter(row.status for row in PT_T01_MATRIX)
    assert counts["ACTIVE"] == 14
    assert counts["HELD_OUT"] == 3
    assert counts["CONDITIONAL"] == 1


def test_pt_t01_held_out_ids() -> None:
    held = {row.nt_id for row in PT_T01_MATRIX if row.status == "HELD_OUT"}
    assert held == {"NT-04b", "NT-07", "NT-15e"}


def test_pt_t01_unknown_disposition_rejected() -> None:
    for row in PT_T01_MATRIX:
        assert row.status in _KNOWN_STATUSES


def test_a4_006_emitter_gate_listed() -> None:
    assert A4_006_TRACKED in ACTIVE_TRACKED_PYTEST_ARGS


@pytest.mark.parametrize("pytest_arg", ACTIVE_TRACKED_PYTEST_ARGS)
def test_pt_t01_active_tracked_source_file_exists(pytest_arg: str) -> None:
    assert (_REPO_ROOT / pytest_arg.split("::", 1)[0]).is_file()


def test_pt_t01_mapped_source_files_exist() -> None:
    for row in PT_T01_MATRIX:
        for path in row.tracked_pytest:
            assert (_REPO_ROOT / path.split("::", 1)[0]).is_file(), path
    assert (_REPO_ROOT / A4_006_TRACKED).is_file()
