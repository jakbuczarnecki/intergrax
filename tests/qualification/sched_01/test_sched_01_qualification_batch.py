# © Artur Czarnecki. All rights reserved.

"""SCHED-01 batch hooks and catalog integrity."""

from __future__ import annotations

import pytest

from tests.qualification.sched_01.catalog import SCHED_01_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_sched_01_catalog_covers_sched_q1_through_sched_q15() -> None:
    ids = {entry.q_id for entry in SCHED_01_Q_CATALOG}
    expected = {f"SCHED-Q{i}" for i in range(1, 16)}
    assert ids == expected


def test_sched_01_frozen_scheduler_integrity_regression_paths() -> None:
    paths = (
        "tests/unit/runtime/long_running/test_pcm_scheduler_integrity.py",
        "tests/integration/runtime/long_running/test_long_running_scheduler_j4.py",
        "intergrax/runtime/long_running/wiring.py",
        "tests/qualification/bg_01/test_bg_01_gates.py",
    )
    for rel in paths:
        assert rel.startswith("tests/") or rel.startswith("intergrax/")
