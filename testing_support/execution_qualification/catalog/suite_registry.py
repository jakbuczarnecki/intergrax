# © Artur Czarnecki. All rights reserved.

"""Global pytest argument vector → suite_id registry (fail-closed on conflicts)."""

from __future__ import annotations

from collections.abc import Mapping

from testing_support.execution_qualification.catalog.labels import (
    NPSC5E_R2_MANDATORY_LABEL_TO_SUITE_ID,
    NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID,
    NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID,
    NPSC5F_R2_DIRECT_LABEL_TO_SUITE_ID,
    NPSC5F_R3_DIRECT_LABEL_TO_SUITE_ID,
    NPSC5F_R4_LABEL_TO_SUITE_ID,
    SHARED_NPSC5D_SUITE_ID,
    shared_suite_id_overrides,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_R2_FINAL_MANDATORY,
    NPSC5E_R3_FINAL_MANDATORY,
    NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
    NPSC5F_R1_FINAL_MANDATORY,
    NPSC5F_R2_FINAL_MANDATORY,
    NPSC5F_R3_FINAL_MANDATORY,
    NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
)
from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
)
from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)

NPSC5F_FINAL_EXTRA_LABEL_TO_SUITE_ID: dict[str, str] = {
    "Recovery": "npsc5f-final.recovery",
    "NPSC-5E Final": "npsc5f-final.npsc-5e-orchestrator",
    "HITL R3": "npsc5e-r3.mandatory.hitl-r3",
    "Child execution": "npsc5e-r3.mandatory.child-execution",
    "Checkpoint": "npsc5f-final.checkpoint",
    "Retry": "npsc5e-r3.mandatory.r1-final",
    "Cancellation": "npsc5f-final.cancellation",
    "Evidence": "npsc5f-final.evidence",
    "NPSC-5D Final": SHARED_NPSC5D_SUITE_ID,
    "NPSC-5F Final drift sentinel": "npsc5f-final.drift-sentinel",
}


def _register_label_map(
    mapping: Mapping[str, str],
    source: FrozenPytestSuiteSource,
    registry: dict[tuple[str, ...], str],
    *,
    skip_existing: bool = False,
) -> None:
    for label, suite_id in mapping.items():
        for src_label, targets in source:
            if src_label != label:
                continue
            key = normalize_pytest_arguments(targets)
            if key in registry:
                if skip_existing:
                    continue
                if registry[key] != suite_id:
                    raise ValueError(
                        f"conflicting suite_id for pytest args {key!r}: "
                        f"{registry[key]!r} vs {suite_id!r}",
                    )
                continue
            registry[key] = suite_id


def _build_pytest_to_suite_id_registry() -> dict[tuple[str, ...], str]:
    registry: dict[tuple[str, ...], str] = {}
    overrides = dict(shared_suite_id_overrides())
    r3_map = dict(NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID)
    r3_map.update(overrides)
    _register_label_map(r3_map, NPSC5E_R3_FINAL_MANDATORY, registry)
    r2_map = dict(NPSC5E_R2_MANDATORY_LABEL_TO_SUITE_ID)
    _register_label_map(r2_map, NPSC5E_R2_FINAL_MANDATORY, registry, skip_existing=True)
    _register_label_map(
        NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID,
        NPSC5F_R1_FINAL_MANDATORY,
        registry,
        skip_existing=True,
    )
    _register_label_map(
        NPSC5F_R2_DIRECT_LABEL_TO_SUITE_ID,
        NPSC5F_R2_FINAL_MANDATORY,
        registry,
        skip_existing=True,
    )
    _register_label_map(
        NPSC5F_R3_DIRECT_LABEL_TO_SUITE_ID,
        NPSC5F_R3_FINAL_MANDATORY,
        registry,
        skip_existing=True,
    )
    _register_label_map(
        NPSC5F_R4_LABEL_TO_SUITE_ID,
        NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
        registry,
        skip_existing=True,
    )
    final_map = dict(NPSC5F_R4_LABEL_TO_SUITE_ID)
    final_map.update(NPSC5F_FINAL_EXTRA_LABEL_TO_SUITE_ID)
    _register_label_map(
        final_map,
        NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
        registry,
        skip_existing=True,
    )
    from testing_support.execution_qualification.final_semantic_pytest import (
        NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,
        NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID,
        NPSC5E_R3_FINAL_SEMANTIC_SUITE_ID,
        NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID,
        npsc5e_r2_final_semantic_pytest_arguments,
        npsc5e_r2_h2_q1_semantic_pytest_arguments,
        npsc5e_r3_final_semantic_pytest_arguments,
        npsc5e_r3_implementation_semantic_pytest_arguments,
    )

    registry[npsc5e_r2_h2_q1_semantic_pytest_arguments()] = (
        NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID
    )
    registry[npsc5e_r2_final_semantic_pytest_arguments()] = (
        NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID
    )
    registry[npsc5e_r3_final_semantic_pytest_arguments()] = (
        NPSC5E_R3_FINAL_SEMANTIC_SUITE_ID
    )
    registry[npsc5e_r3_implementation_semantic_pytest_arguments()] = (
        NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID
    )
    return registry


_PYTEST_TO_SUITE_ID: dict[tuple[str, ...], str] = _build_pytest_to_suite_id_registry()


def suite_id_for_pytest_arguments(pytest_arguments: tuple[str, ...]) -> str:
    key = normalize_pytest_arguments(pytest_arguments)
    if key not in _PYTEST_TO_SUITE_ID:
        raise ValueError(f"no suite_id for pytest arguments: {key!r}")
    return _PYTEST_TO_SUITE_ID[key]


def pytest_to_suite_id_registry() -> Mapping[tuple[str, ...], str]:
    return dict(_PYTEST_TO_SUITE_ID)
