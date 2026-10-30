# © Artur Czarnecki. All rights reserved.

"""EBH-7 W4 governance / DUP / SYS-INV revalidation gates."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_w4_dup_x_l2_subset_importable() -> None:
    import tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary as b4

    assert hasattr(b4, "test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone")


def test_ebh7_w4_dup_x_npsc_boundary_gates_importable() -> None:
    import tests.qualification.dup_x.test_dup_x_npsc_boundary_gates as npsc

    assert hasattr(npsc, "test_dup_x_npsc_exclusion_record_covers_ten_modules")
