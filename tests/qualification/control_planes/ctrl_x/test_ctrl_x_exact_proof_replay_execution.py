# © Artur Czarnecki. All rights reserved.

"""CTRL-X — execute catalog proof nodes (non-recursive)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.control_planes.ctrl_x.proof_replay import run_ctrl_x_exact_proof_replay

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ctrl_x_exact_proof_replay_all_catalog_nodes_pass() -> None:
    report = run_ctrl_x_exact_proof_replay(_REPO_ROOT)
    assert report.exit_code == 0, (
        f"exit={report.exit_code} failed={report.failed} errors={report.errors} "
        f"skipped={report.skipped} passed={report.executed_passed}/{report.requested}"
    )
    assert report.failed == 0
    assert report.errors == 0
    assert report.xfailed == 0
    assert report.skipped == 0
    assert report.executed_passed == report.requested
