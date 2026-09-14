# © Artur Czarnecki. All rights reserved.

"""Canonical pytest slices for Final modules without embedded qualification harness."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    NPSC5E_R2_FINAL_ORCHESTRATOR_PATH,
    NPSC5E_R3_FINAL_ORCHESTRATOR_PATH,
)
from testing_support.execution_qualification.embedded_harness_kexpr import (
    CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
)

NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID = "npsc5e-r2.final-semantic"
NPSC5E_R3_FINAL_SEMANTIC_SUITE_ID = "npsc5e-r3.final-semantic"


def npsc5e_r2_final_semantic_pytest_arguments() -> tuple[str, ...]:
    return (
        NPSC5E_R2_FINAL_ORCHESTRATOR_PATH,
        "-k",
        CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
    )


def npsc5e_r3_final_semantic_pytest_arguments() -> tuple[str, ...]:
    return (
        NPSC5E_R3_FINAL_ORCHESTRATOR_PATH,
        "-k",
        CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
    )


def npsc5f_final_recovery_pytest_arguments(
    pytest_paths: tuple[str, ...],
) -> tuple[str, ...]:
    return normalize_pytest_arguments(
        [*pytest_paths, "-k", CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR],
    )


def pytest_arguments_exclude_embedded_harness(
    pytest_arguments: tuple[str, ...],
) -> bool:
    normalized = normalize_pytest_arguments(pytest_arguments)
    if "-k" not in normalized:
        return False
    k_index = normalized.index("-k")
    if k_index + 1 >= len(normalized):
        return False
    return normalized[k_index + 1] == CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR


def profile_final_semantic_argument_sets(profile_id: str) -> frozenset[tuple[str, ...]]:
    from testing_support.execution_qualification.catalog.profile_builders import (
        NPSC5E_R2_PROFILE_ID,
        NPSC5E_R3_PROFILE_ID,
    )

    if profile_id == NPSC5E_R2_PROFILE_ID:
        return frozenset({npsc5e_r2_final_semantic_pytest_arguments()})
    if profile_id == NPSC5E_R3_PROFILE_ID:
        return frozenset({npsc5e_r3_final_semantic_pytest_arguments()})
    return frozenset()
