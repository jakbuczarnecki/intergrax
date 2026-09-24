# © Artur Czarnecki. All rights reserved.

"""Static guards — UCA-6C-R5.9-R1-R1 forbids scalar/payload restart handoffs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_E2E = (
    _REPO
    / "tests"
    / "unit"
    / "autonomous_work"
    / "test_uca6c_r6_r5_9_r1_true_restart_worker_governed_e2e.py"
)
_BACKENDS = (
    _REPO / "testing_support" / "uca6c_r6_r5_9_r1_true_restart_durable_backends.py"
)


def test_true_restart_e2e_forbids_scalar_run_id_registration_handoff() -> None:
    source = _E2E.read_text(encoding="utf-8")
    forbidden = [
        "ActiveTaskRegistry.register(task_b, scalars.run_id)",
        "replace(host_c.stack, run_id=scalars.run_id)",
    ]
    for pattern in forbidden:
        assert pattern not in source


def test_true_restart_backends_forbids_continuation_export_field() -> None:
    source = _BACKENDS.read_text(encoding="utf-8")
    assert "continuation_export" not in source
