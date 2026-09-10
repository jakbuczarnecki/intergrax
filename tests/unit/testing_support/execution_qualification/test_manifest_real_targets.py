# © Artur Czarnecki. All rights reserved.

"""Prove typed manifest accepts real gate-style pytest targets without running full matrix."""

from __future__ import annotations

from testing_support.execution_qualification.contracts import QualificationRunManifest, QualificationSuite
from testing_support.execution_qualification.executor import build_pytest_command


def test_representative_execution_gate_target_shape() -> None:
    suite = QualificationSuite(
        suite_id="P0A-style",
        pytest_arguments=(
            "tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py",
            "-q",
            "--tb=no",
        ),
    )
    manifest = QualificationRunManifest(suites=(suite,))
    command = build_pytest_command(manifest.suites[0])
    assert command[0:3] == ("uv", "run", "pytest")
    assert "test_npsc5e_p0a" in command[3]


def test_exclusive_resource_manifest_field() -> None:
    suite = QualificationSuite(
        suite_id="impl-gate-style",
        pytest_arguments=(
            "tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py",
            "-q",
            "--tb=no",
        ),
        exclusive_resource_id=".tmp/session/npsc5e-r3/cross.db",
    )
    assert suite.exclusive_resource_id is not None
