# © Artur Czarnecki. All rights reserved.

"""F09 adversarial fail-closed tests for finding resolution registry."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._gate_finding_resolution import (
    FindingResolution,
    FindingResolutionKind,
    validate_finding_resolution,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_f09_missing_resolution_record_fails() -> None:
    res = FindingResolution("GHOST-ID", FindingResolutionKind.CORRECTIVE_PYTEST, "tests/x.py::test_nope")
    err = validate_finding_resolution(res)
    assert err and "missing" in err


def test_ebh7_f09_nonexistent_pytest_node_fails() -> None:
    res = FindingResolution(
        "PROBE-1",
        FindingResolutionKind.CORRECTIVE_PYTEST,
        "tests/qualification/ebh_7/test_ebh_7_gate_finding_resolution_adversarial.py::test_this_function_does_not_exist",
    )
    err = validate_finding_resolution(res)
    assert err and "missing" in err


def test_ebh7_f09_pytest_outside_l3_profile_fails() -> None:
    res = FindingResolution(
        "PROBE-2",
        FindingResolutionKind.CORRECTIVE_PYTEST,
        "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py::test_b6_05_prohibited_unresolved_nonzero",
    )
    err = validate_finding_resolution(res)
    assert err and "L3 MANIFEST profile" in err


def test_ebh7_f09_bad_assert_hook_fails() -> None:
    res = FindingResolution(
        "PROBE-3",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._gate_finding_resolution:validate_finding_resolution",
    )
    err = validate_finding_resolution(res)
    assert err and ("failed" in err or "error" in err)


def test_ebh7_f09_supersession_missing_anchor_fails() -> None:
    res = FindingResolution(
        "PROBE-4",
        FindingResolutionKind.HISTORICAL_SUPERSEDED,
        "docs/project/maintainers/qualification/EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md@missing-anchor-xyz",
        notes="probe",
    )
    err = validate_finding_resolution(res)
    assert err and "anchor" in err


def test_ebh7_f09_registry_integrity_passes() -> None:
    from tests.qualification.ebh_7._gate_finding_resolution import assert_ebh7_f09_resolution_registry_integrity

    assert_ebh7_f09_resolution_registry_integrity()


def test_ebh7_f07_dual_slug_matrix_complete() -> None:
    from tests.qualification.ebh_5._dual_slug_rpl_ef_matrix import assert_dual_slug_rpl_ef_matrix_complete

    assert_dual_slug_rpl_ef_matrix_complete()
