# © Artur Czarnecki. All rights reserved.

"""SSOT pytest -k expression excluding embedded qualification harness tests."""

from __future__ import annotations

R2_H2_Q1_EMBEDDED_HARNESS_KEXPR: str = "not test_mandatory_frozen_suite_passes"

CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR: str = (
    "not test_mandatory_frozen_suite_passes and "
    "not test_mandatory_frozen_suites_pass_via_parallel_qualification and "
    "not test_mandatory_regression_matrix_passes and "
    "not test_r4_mandatory_regression_matrix and "
    "not test_npsc5f_final_mandatory_regression_matrix_passes and "
    "not test_pre_existing_cancellation_fixture_same_root_cause and "
    "not test_pre_existing_partial_results_fixture_unchanged_baseline and "
    "not test_ruff_recovery_surfaces_and_final_test and "
    "not test_pyright_recovery_surfaces_and_final_test and "
    "not test_pre_existing_cancellation_fixture_invalid_persist_gate and "
    "not test_pre_existing_partial_results_unrelated_to_r2 and "
    "not test_ruff_final_test_no_new_errors and "
    "not test_pyright_final_test_no_new_errors and "
    "not test_ruff_r3_surface_and_final_test_no_new_errors and "
    "not test_pyright_r3_surface_and_final_test_no_new_errors"
)

_EMBEDDED_HARNESS_TEST_NAMES: frozenset[str] = frozenset(
    {
        "test_mandatory_frozen_suite_passes",
        "test_mandatory_frozen_suites_pass_via_parallel_qualification",
        "test_mandatory_regression_matrix_passes",
        "test_r4_mandatory_regression_matrix",
        "test_npsc5f_final_mandatory_regression_matrix_passes",
    },
)

_EMBEDDED_HARNESS_CALL_NAMES: frozenset[str] = frozenset(
    {
        "run_npsc5e_r3_mandatory_qualification",
        "validate_and_run_measured",
    },
)


def embedded_harness_test_names() -> frozenset[str]:
    return _EMBEDDED_HARNESS_TEST_NAMES


def embedded_harness_call_names() -> frozenset[str]:
    return _EMBEDDED_HARNESS_CALL_NAMES


def pytest_k_expression_excludes_embedded_harness(k_expr: str) -> bool:
    return "test_mandatory_frozen_suite_passes" in k_expr and (
        k_expr == CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR
        or k_expr == R2_H2_Q1_EMBEDDED_HARNESS_KEXPR
    )
