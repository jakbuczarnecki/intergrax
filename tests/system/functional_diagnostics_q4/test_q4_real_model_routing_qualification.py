# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q4 qualification semantics regression."""

from __future__ import annotations

import pytest

from tests.system.functional_diagnostics_q4.cases import (
    Q4_A_HEALTHY,
    Q4_B_WRONG_ROUTE,
    Q4_C_INVOKE_FAILURE,
    Q4_D_VALIDATION_FAILURE,
    Q4_E_MISSING_EVIDENCE,
)
from intergrax.runtime.diagnostics.specifications.q4_model_routing_functional_diagnostic_specification import (
    CHECK_Q4_INVOCATION,
    CHECK_Q4_SELECTION,
    CHECK_Q4_VALIDATION,
)

pytestmark = pytest.mark.unit


def test_q4_b_first_failure_is_selection() -> None:
    assert Q4_B_WRONG_ROUTE.expected_first_proven_failed_check == CHECK_Q4_SELECTION


def test_q4_c_first_failure_is_invocation() -> None:
    assert Q4_C_INVOKE_FAILURE.expected_first_proven_failed_check == CHECK_Q4_INVOCATION


def test_q4_d_first_failure_is_validation() -> None:
    assert Q4_D_VALIDATION_FAILURE.expected_first_proven_failed_check == CHECK_Q4_VALIDATION


def test_q4_e_is_inconclusive_without_validation() -> None:
    assert Q4_E_MISSING_EVIDENCE.include_validation is False
    assert Q4_E_MISSING_EVIDENCE.expected_first_proven_failed_check is None


def test_q4_a_is_full_success() -> None:
    assert Q4_A_HEALTHY.expected_first_proven_failed_check is None
