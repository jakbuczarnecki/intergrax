# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R2 qualification gates."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_persisted_r2_r2_gates import (
    compute_persisted_r2_r2_metrics,
    inventory_matches_legacy_obligations,
    persisted_r2_r2_gate_passes,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_r2_r2_gate_passes() -> None:
    assert persisted_r2_r2_gate_passes()
    assert inventory_matches_legacy_obligations()


def test_cx_r2_r2_metrics_closed_world() -> None:
    metrics = compute_persisted_r2_r2_metrics()
    assert metrics.unclassified_legacy_obligations == 0
    assert metrics.migration_owner_missing == 0
    assert metrics.duplicate_migration_authority == 0
    assert metrics.implicit_compatibility_branches == 0
