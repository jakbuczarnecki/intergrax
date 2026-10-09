# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1-R1-R1 configured execution subject architecture gates (docs-only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_START_HEAD = "f517fd787d0e10488eed9a0da97c1ea1c5166f0c"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_CONFIGURED_EXECUTION_SUBJECT_ARCHITECTURE_LOCK.md"
)
_R1_R1_HANDOFF = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_CANONICAL_CONFIGURED_EXECUTION_HANDOFF.md"
)
_QUALIFIED_DISPATCH = (
    _REPO_ROOT
    / "intergrax/contracts/execution/qualified_capability_execution_dispatch.py"
)
_EXECUTION_BOUND_DISPATCH = (
    _REPO_ROOT
    / "intergrax/contracts/execution/execution_bound_capability_execution_dispatch.py"
)
_CANDIDATE_CONTRACT = (
    _REPO_ROOT / "intergrax/contracts/autonomous_work/capability_acquisition.py"
)


def test_txp5r2p3r1r1r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r1r1r1_q02_architecture_lock_exists_ready_for_audit() -> None:
    assert _LOCK_DOC.is_file()
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in text
    assert "ConfiguredCapabilityExecutionSubject" in text
    assert "FRZ-TRC-11" in text and "OPEN" in text


def test_txp5r2p3r1r1r1_q03_r1_r1_q3_marked_rejected_superseded() -> None:
    handoff = _R1_R1_HANDOFF.read_text(encoding="utf-8")
    assert "SUPERSEDED / REJECTED" in handoff
    assert "TRACE_X_P5_R2_P3_R1_R1_R1_CONFIGURED_EXECUTION_SUBJECT_ARCHITECTURE_LOCK" in handoff
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "R1-R1 Q3" in lock or "R1-R1 because" in lock
    assert "REJECT" in lock
    assert "reuse acquisition_result" in lock or "reuse qualification_result" in lock


def test_txp5r2p3r1r1r1_q04_qualified_dispatch_requires_uca_ids() -> None:
    source = _QUALIFIED_DISPATCH.read_text(encoding="utf-8")
    assert "qualification_request_id: str" in source
    assert "acquisition_request_id: str" in source


def test_txp5r2p3r1r1r1_q05_execution_bound_dispatch_has_no_uca_qualification_ids() -> None:
    source = _EXECUTION_BOUND_DISPATCH.read_text(encoding="utf-8")
    assert "direct_reuse_operation_id" in source
    assert "qualification_request_id" not in source
    assert "acquisition_request_id" not in source


def test_txp5r2p3r1r1r1_q06_candidate_has_no_execution_target_field() -> None:
    source = _CANDIDATE_CONTRACT.read_text(encoding="utf-8")
    assert "class WorkerCapabilityCandidate" in source
    assert "QualifiedCapabilityExecutionTarget" not in source
    assert "business_execution_target" not in source


def test_txp5r2p3r1r1r1_q07_lock_rejects_uca_reuse_and_string_target() -> None:
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "CONFIGURE_EXISTING ∩ CapabilityGap" in text or "∩" in text
    assert "capability_ref" in text
    assert "REJECT" in text
    assert "Option A" in text
    assert "Production delta" in text and "**0**" in text
