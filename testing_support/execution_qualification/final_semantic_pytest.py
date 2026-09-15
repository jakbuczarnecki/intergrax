# © Artur Czarnecki. All rights reserved.

"""Canonical pytest slices for Final modules without embedded qualification harness."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_R2_FINAL_EMBEDDED_PREDECESSOR_LABELS,
    NPSC5E_R2_FINAL_MANDATORY,
    NPSC5E_R2_H2_Q1_EMBEDDED_PREDECESSOR_LABELS,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_R3_FINAL_MANDATORY,
    NPSC5E_R3_IMPLEMENTATION_EMBEDDED_PREDECESSOR_LABELS,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    NPSC5E_R2_FINAL_ORCHESTRATOR_PATH,
    NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH,
    NPSC5E_R3_FINAL_ORCHESTRATOR_PATH,
    NPSC5E_R3_IMPLEMENTATION_ORCHESTRATOR_PATH,
)
from testing_support.execution_qualification.embedded_harness_kexpr import (
    CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
    R2_H2_Q1_EMBEDDED_HARNESS_KEXPR,
)

NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID = "npsc5e-r2.final-semantic"
NPSC5E_R3_FINAL_SEMANTIC_SUITE_ID = "npsc5e-r3.final-semantic"
NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID = "npsc5e-r3.implementation-semantic"
NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID = "npsc5e-r2.mandatory.r2-h2-q1"


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


def npsc5e_r3_implementation_semantic_pytest_arguments() -> tuple[str, ...]:
    return (
        NPSC5E_R3_IMPLEMENTATION_ORCHESTRATOR_PATH,
        "-k",
        CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
    )


def npsc5e_r2_h2_q1_semantic_pytest_arguments() -> tuple[str, ...]:
    return (
        NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH,
        "-k",
        R2_H2_Q1_EMBEDDED_HARNESS_KEXPR,
    )


_R2_H2_Q1_EMBEDDED_LABEL_ALIASES: dict[str, str] = {
    "DG_001 lineage": "DG_001",
}

_R3_IMPLEMENTATION_EMBEDDED_LABEL_ALIASES: dict[str, str] = {
    "NPSC-5B": "NPSC-5B Final",
}


def npsc5e_r2_final_embedded_predecessor_suite_ids() -> tuple[str, ...]:
    from testing_support.execution_qualification.catalog.suite_registry import (
        suite_id_for_pytest_arguments,
    )

    mandatory_by_label = dict(NPSC5E_R2_FINAL_MANDATORY)
    suite_ids: list[str] = []
    seen: set[str] = set()
    for label in NPSC5E_R2_FINAL_EMBEDDED_PREDECESSOR_LABELS:
        targets = mandatory_by_label[label]
        suite_id = suite_id_for_pytest_arguments(normalize_pytest_arguments(targets))
        if suite_id in seen:
            raise ValueError(
                f"ambiguous predecessor resolution for R2 Final label {label!r}: "
                f"{suite_id!r}",
            )
        seen.add(suite_id)
        suite_ids.append(suite_id)
    return tuple(suite_ids)


def npsc5e_r3_implementation_embedded_predecessor_suite_ids() -> tuple[str, ...]:
    from testing_support.execution_qualification.catalog.suite_registry import (
        suite_id_for_pytest_arguments,
    )

    mandatory_by_label = dict(NPSC5E_R3_FINAL_MANDATORY)
    suite_ids: list[str] = []
    seen: set[str] = set()
    for label in NPSC5E_R3_IMPLEMENTATION_EMBEDDED_PREDECESSOR_LABELS:
        canonical_label = _R3_IMPLEMENTATION_EMBEDDED_LABEL_ALIASES.get(label, label)
        targets = mandatory_by_label[canonical_label]
        suite_id = suite_id_for_pytest_arguments(normalize_pytest_arguments(targets))
        if suite_id in seen:
            raise ValueError(
                f"ambiguous predecessor resolution for R3 implementation label "
                f"{label!r}: {suite_id!r}",
            )
        seen.add(suite_id)
        suite_ids.append(suite_id)
    return tuple(suite_ids)


def npsc5e_r2_h2_q1_embedded_predecessor_suite_ids() -> tuple[str, ...]:
    from testing_support.execution_qualification.catalog.suite_registry import (
        suite_id_for_pytest_arguments,
    )

    mandatory_by_label = dict(NPSC5E_R2_FINAL_MANDATORY)
    suite_ids: list[str] = []
    seen: set[str] = set()
    for label in NPSC5E_R2_H2_Q1_EMBEDDED_PREDECESSOR_LABELS:
        canonical_label = _R2_H2_Q1_EMBEDDED_LABEL_ALIASES.get(label, label)
        targets = mandatory_by_label[canonical_label]
        suite_id = suite_id_for_pytest_arguments(normalize_pytest_arguments(targets))
        if suite_id in seen:
            continue
        seen.add(suite_id)
        suite_ids.append(suite_id)
    return tuple(suite_ids)


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
    k_expr = normalized[k_index + 1]
    return k_expr in (
        CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
        R2_H2_Q1_EMBEDDED_HARNESS_KEXPR,
    )


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
