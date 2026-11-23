# © Artur Czarnecki. All rights reserved.
from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.campaign_b._pt_t01_matrix import (
    A4_006_TRACKED,
    ACTIVE_TRACKED_PYTEST_ARGS,
    EXPECTED_NT_IDS,
    PT_T01_MATRIX,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pt_t01_matrix_covers_nt01_through_nt16() -> None:
    assert {row.nt_id for row in PT_T01_MATRIX} == EXPECTED_NT_IDS


def test_pt_t01_held_out_have_no_tracked_tests() -> None:
    for row in PT_T01_MATRIX:
        if row.status == "HELD_OUT":
            assert not row.tracked_pytest


@pytest.mark.parametrize("pytest_arg", ACTIVE_TRACKED_PYTEST_ARGS)
def test_pt_t01_active_tracked_node_collectable(pytest_arg: str) -> None:
    assert (_REPO_ROOT / pytest_arg.split("::", 1)[0]).is_file()


def test_pt_t01_active_tracked_args_non_empty() -> None:
    assert len(ACTIVE_TRACKED_PYTEST_ARGS) >= 16


def test_a4_006_emitter_gate_listed() -> None:
    assert A4_006_TRACKED in ACTIVE_TRACKED_PYTEST_ARGS
