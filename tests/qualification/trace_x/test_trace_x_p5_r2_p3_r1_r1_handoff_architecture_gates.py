# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1-R1 mechanical handoff architecture gates (docs-only child)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_START_HEAD = "61cfc96a42cef104819b90ca36a7dab7decfeba1"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/TRACE_X_P5_R2_P3_R1_R1_CANONICAL_CONFIGURED_EXECUTION_HANDOFF.md"
)
_FULFILLMENT = (
    _REPO_ROOT / "intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py"
)


def test_txp5r2p3r1r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r1r1_q02_architecture_lock_document_exists() -> None:
    assert _LOCK_DOC.is_file()
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in text
    assert "worker_acquisition_decision" in text
    assert "TO BE REMOVED / NOT SANCTIONED" in text


def test_txp5r2p3r1r1_q03_recovery_coordinator_does_not_emit_configure_existing_required() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/autonomous_work/worker_capability_recovery_coordinator.py"
    ).read_text(encoding="utf-8")
    assert "CONFIGURE_EXISTING_REQUIRED" not in source


def test_txp5r2p3r1r1_q04_fulfillment_request_has_no_decision_field() -> None:
    source = (
        _REPO_ROOT / "intergrax/contracts/autonomous_work/worker_capability_fulfillment.py"
    ).read_text(encoding="utf-8")
    assert "WorkerCapabilityAcquisitionDecision" not in source


def test_txp5r2p3r1r1_q05_option_a_phase_gate_debt_documented_in_code() -> None:
    source = _FULFILLMENT.read_text(encoding="utf-8")
    assert "QUALIFICATION_COMPLETE" in source
    assert "_fulfill_configure_existing" in source
