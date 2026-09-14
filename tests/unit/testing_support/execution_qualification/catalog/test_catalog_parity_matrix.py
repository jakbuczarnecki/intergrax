# © Artur Czarnecki. All rights reserved.

"""Target-set parity between legacy expansion and canonical catalog profiles."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.expansion import (
    unique_required_leaf_targets,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_FINAL_MANDATORY,
    NPSC5E_R1_FINAL_MANDATORY,
    NPSC5E_R2_FINAL_MANDATORY,
    NPSC5E_R3_FINAL_MANDATORY,
    NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
    NPSC5F_R1_FINAL_MANDATORY,
    NPSC5F_R2_FINAL_MANDATORY,
    NPSC5F_R3_FINAL_MANDATORY,
    NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_FINAL_PROFILE_ID,
    NPSC5E_R1_PROFILE_ID,
    NPSC5E_R2_PROFILE_ID,
    NPSC5E_R3_PROFILE_ID,
    NPSC5F_FINAL_PROFILE_ID,
    NPSC5F_R1_PROFILE_ID,
    NPSC5F_R2_PROFILE_ID,
    NPSC5F_R3_PROFILE_ID,
    NPSC5F_R4_PROFILE_ID,
)
from testing_support.npsc5f_r1_qualification_profile import dag_required_target_set

_PROFILE_TO_MANDATORY: dict[str, object] = {
    NPSC5F_R1_PROFILE_ID: NPSC5F_R1_FINAL_MANDATORY,
    NPSC5E_R3_PROFILE_ID: NPSC5E_R3_FINAL_MANDATORY,
    NPSC5E_R2_PROFILE_ID: NPSC5E_R2_FINAL_MANDATORY,
    NPSC5E_R1_PROFILE_ID: NPSC5E_R1_FINAL_MANDATORY,
    NPSC5E_FINAL_PROFILE_ID: NPSC5E_FINAL_MANDATORY,
    NPSC5F_R2_PROFILE_ID: NPSC5F_R2_FINAL_MANDATORY,
    NPSC5F_R3_PROFILE_ID: NPSC5F_R3_FINAL_MANDATORY,
    NPSC5F_R4_PROFILE_ID: NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
    NPSC5F_FINAL_PROFILE_ID: NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
}


def _legacy_leaf_arg_set(
    mandatory: tuple[tuple[str, list[str]], ...],
) -> frozenset[tuple[str, ...]]:
    return frozenset(
        entry.pytest_arguments for entry in unique_required_leaf_targets(mandatory)
    )


def _assert_parity(
    profile_id: str, mandatory: tuple[tuple[str, list[str]], ...]
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(profile_id)
    legacy_set = _legacy_leaf_arg_set(mandatory)
    dag_set = dag_required_target_set(compiled.plan)
    assert legacy_set == dag_set, profile_id


def test_parity_npsc5f_r1_final() -> None:
    _assert_parity(NPSC5F_R1_PROFILE_ID, NPSC5F_R1_FINAL_MANDATORY)


def test_parity_npsc5e_r3_final() -> None:
    _assert_parity(NPSC5E_R3_PROFILE_ID, NPSC5E_R3_FINAL_MANDATORY)


def test_parity_npsc5e_r2_final() -> None:
    _assert_parity(NPSC5E_R2_PROFILE_ID, NPSC5E_R2_FINAL_MANDATORY)


def test_parity_npsc5e_r1_final() -> None:
    _assert_parity(NPSC5E_R1_PROFILE_ID, NPSC5E_R1_FINAL_MANDATORY)


def test_parity_npsc5e_final() -> None:
    _assert_parity(NPSC5E_FINAL_PROFILE_ID, NPSC5E_FINAL_MANDATORY)


def test_parity_npsc5f_r2_final() -> None:
    _assert_parity(NPSC5F_R2_PROFILE_ID, NPSC5F_R2_FINAL_MANDATORY)


def test_parity_npsc5f_r3_final() -> None:
    _assert_parity(NPSC5F_R3_PROFILE_ID, NPSC5F_R3_FINAL_MANDATORY)


def test_parity_npsc5f_r4_final() -> None:
    _assert_parity(NPSC5F_R4_PROFILE_ID, NPSC5F_R4_MANDATORY_REGRESSION_SUITES)


def test_parity_npsc5f_final() -> None:
    _assert_parity(NPSC5F_FINAL_PROFILE_ID, NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES)


def test_catalog_profile_compile_is_deterministic() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in catalog.profile_ids:
        first = catalog.compile_profile(profile_id)
        second = catalog.compile_profile(profile_id)
        assert first.plan == second.plan
