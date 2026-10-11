# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R1-R1 qualification gates (idempotency ledger schema closure)."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_persisted_r2_r1_gates import (
    build_persisted_r2_r1_qualification_report,
    persisted_r2_r1_inventory_gate_passes,
)
from tests.qualification.compat_x._compat_x_persisted_r2_r1_r1_gates import (
    build_idempotency_schema_gate_report,
    persisted_r2_r1_r1_schema_gate_passes,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_r2_r1_r1_idempotency_schema_gate() -> None:
    assert persisted_r2_r1_r1_schema_gate_passes()


def test_cx_r2_r1_r1_r2_r1_continuity() -> None:
    assert persisted_r2_r1_inventory_gate_passes()
    report = build_persisted_r2_r1_qualification_report()
    assert report.missing_version_semantic_family_count == 0
    assert report.opaque_carrier_family_count >= 1


def test_cx_r2_r1_r1_source_evidence_report() -> None:
    gate = build_idempotency_schema_gate_report()
    assert gate.sqlite_mutation_methods_gated
    assert gate.redis_lua_scripts_gated
    assert gate.redis_external_effect_read_gated
