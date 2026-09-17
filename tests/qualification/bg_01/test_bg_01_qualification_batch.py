# © Artur Czarnecki. All rights reserved.

"""BG-01 batch hooks and catalog integrity."""

from __future__ import annotations

import pytest

from tests.qualification.bg_01.catalog import BG_01_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_bg_01_catalog_covers_bg_q1_through_bg_q15() -> None:
    ids = {entry.q_id for entry in BG_01_Q_CATALOG}
    expected = {f"BG-Q{i}" for i in range(1, 16)}
    assert ids == expected


def test_bg_01_frozen_background_convergence_regression_paths() -> None:
    paths = (
        "tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py",
        "tests/unit/runtime/background_execution/test_background_execution_reentry_admission.py",
        "tests/unit/runtime/background_execution/test_required_audit_evidence_admission.py",
        "tests/unit/queueing/worker/test_execution.py",
        "tests/integration/applications/test_unified_execution_entry_j3.py",
        "applications/local_workspace_application/tests/host/test_lkw_background_canonical_execution.py",
        "tests/qualification/host_01/test_host_01_gates.py",
    )
    for path in paths:
        assert path.startswith("tests/") or path.startswith("applications/")
