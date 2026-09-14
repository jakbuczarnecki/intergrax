# © Artur Czarnecki. All rights reserved.

"""Legacy regression matrix adapters consume canonical mandatory sources."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
    NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
)
from testing_support.npsc5f_final_regression_matrix import (
    MANDATORY_REGRESSION_SUITES as FINAL_MATRIX_SUITES,
)
from testing_support.npsc5f_r4_regression_matrix import (
    MANDATORY_REGRESSION_SUITES as R4_MATRIX_SUITES,
)


def test_legacy_r4_matrix_consumes_canonical_source() -> None:
    assert R4_MATRIX_SUITES == NPSC5F_R4_MANDATORY_REGRESSION_SUITES


def test_legacy_final_matrix_consumes_canonical_source() -> None:
    assert FINAL_MATRIX_SUITES == NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES
