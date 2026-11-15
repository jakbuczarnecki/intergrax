# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-B pinned tests: F-D2-10 the static pre-context policy wiring gate follows the relocated UAEP."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from intergrax.runtime.policy.pre_context_policy_audit import audit_pre_context_policy_wiring

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_MARKERS = "HookPoint.BEFORE_CONTEXT_BUILD\nRuntimeEventType.POLICY_DECISION\n"


def test_f_d2_10_the_audit_finds_its_scan_roots_in_the_repository() -> None:
    assert audit_pre_context_policy_wiring(_REPO) == []


def test_f_d2_10_the_gate_script_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "maintenance" / "check_pre_context_policy_wiring.py")],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_f_d2_10_an_empty_tree_fails_closed(tmp_path: Path) -> None:
    missing = audit_pre_context_policy_wiring(tmp_path)
    assert missing and "missing scan root" in missing[0]


def test_f_d2_10_markers_without_the_relocated_uaep_root_fail_closed(tmp_path: Path) -> None:
    runtime = tmp_path / "intergrax" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "gate.py").write_text(_MARKERS, encoding="utf-8")

    missing = audit_pre_context_policy_wiring(tmp_path)

    assert missing and "missing scan root" in missing[0]


def test_f_d2_10_markers_in_the_relocated_tree_pass(tmp_path: Path) -> None:
    uaep = tmp_path / "intergrax" / "runtime" / "nexus" / "uaep"
    uaep.mkdir(parents=True)
    (uaep / "uaep_executor.py").write_text(_MARKERS, encoding="utf-8")

    assert audit_pre_context_policy_wiring(tmp_path) == []


def test_f_d2_10_missing_markers_are_reported(tmp_path: Path) -> None:
    uaep = tmp_path / "intergrax" / "runtime" / "nexus" / "uaep"
    uaep.mkdir(parents=True)
    (uaep / "uaep_executor.py").write_text("x = 1\n", encoding="utf-8")

    assert set(audit_pre_context_policy_wiring(tmp_path)) == {
        "HookPoint.BEFORE_CONTEXT_BUILD",
        "RuntimeEventType.POLICY_DECISION",
    }
