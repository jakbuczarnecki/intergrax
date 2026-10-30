# © Artur Czarnecki. All rights reserved.

"""EBH-7 structural SSOT exit (not independent acceptance)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._mechanism_registry import assert_ebh7_mechanism_registry_integrity
from tests.qualification.ebh_7._task_register import assert_ebh7_task_register_integrity
from tests.qualification.ebh_7._traceability_matrix import assert_ebh7_traceability_integrity

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_structural_exit_traceability_and_mechanisms() -> None:
    assert_ebh7_traceability_integrity()
    assert_ebh7_task_register_integrity()
    assert_ebh7_mechanism_registry_integrity()


def test_ebh7_structural_exit_npsc_gate_executes() -> None:
    from tests.unit.runtime.architecture.test_npsc4_2_residual_compatibility_gate import (
        test_npsc42_orchestration_backend_access_confined_to_allowlist,
    )

    test_npsc42_orchestration_backend_access_confined_to_allowlist()
