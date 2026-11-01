# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.selection.oracle import (
    FAULT_INJECTION_MATRIX,
    SelectionFaultClass,
    expected_status_for_fault,
)

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_selection_fault_injection_matrix() -> None:
    assert len(FAULT_INJECTION_MATRIX) >= 8
    assert expected_status_for_fault(SelectionFaultClass.DIGEST_MISMATCH) is SelectionStatus.REFUSED_INTEGRITY
    assert expected_status_for_fault(SelectionFaultClass.UNKNOWN_PATH) is SelectionStatus.CONSERVATIVE_FULL
