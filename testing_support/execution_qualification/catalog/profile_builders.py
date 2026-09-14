# © Artur Czarnecki. All rights reserved.

"""Compile canonical qualification profiles from catalog mandatory sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from testing_support.execution_qualification.catalog.expansion import (
    CatalogRequiredTarget,
    unique_required_leaf_targets,
)
from testing_support.execution_qualification.catalog.labels import (
    NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID,
    NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID,
)
from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
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
from testing_support.execution_qualification.catalog.suite_registry import (
    suite_id_for_pytest_arguments,
)
from testing_support.execution_qualification.compiler import (
    compile_qualification_execution_plan,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunManifest,
    QualificationSuite,
)
from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationGateDefinition,
    QualificationGraphDefinition,
    QualificationProfile,
)
from testing_support.execution_qualification.catalog.contracts import (
    CompiledCatalogProfile,
)

NPSC5F_R1_PROFILE_ID = "npsc5f-r1-final"
NPSC5F_R1_ROOT_GATE_ID = "npsc5f-r1.final"

NPSC5E_R3_PROFILE_ID = "npsc5e-r3-final"
NPSC5E_R2_PROFILE_ID = "npsc5e-r2-final"
NPSC5E_R1_PROFILE_ID = "npsc5e-r1-final"
NPSC5E_FINAL_PROFILE_ID = "npsc5e-final"
NPSC5F_R2_PROFILE_ID = "npsc5f-r2-final"
NPSC5F_R3_PROFILE_ID = "npsc5f-r3-final"
NPSC5F_R4_PROFILE_ID = "npsc5f-r4-final"
NPSC5F_FINAL_PROFILE_ID = "npsc5f-final"


def _exclusive_resource_for_suite_id(suite_id: str) -> str | None:
    if suite_id == NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID["R3 implementation gate"]:
        return NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID
    return None


def _build_suites_from_leaves(
    leaves: tuple[CatalogRequiredTarget, ...],
) -> tuple[QualificationSuite, ...]:
    suites: list[QualificationSuite] = []
    seen: set[str] = set()
    for entry in leaves:
        suite_id = suite_id_for_pytest_arguments(entry.pytest_arguments)
        if suite_id in seen:
            continue
        seen.add(suite_id)
        suites.append(
            QualificationSuite(
                suite_id=suite_id,
                pytest_arguments=entry.pytest_arguments,
                exclusive_resource_id=_exclusive_resource_for_suite_id(suite_id),
            ),
        )
    return tuple(suites)


def _gate_id_for_suite(suite_id: str, branch: str) -> str:
    return f"{branch}.requires.{suite_id}"


def _flat_profile(
    *,
    profile_id: str,
    root_gate_id: str,
    mandatory: FrozenPytestSuiteSource,
    branch: str,
) -> CompiledCatalogProfile:
    leaves = unique_required_leaf_targets(mandatory)
    suites = _build_suites_from_leaves(leaves)
    manifest = QualificationRunManifest(suites=suites)
    gate_ids: list[str] = []
    gates: list[QualificationGateDefinition] = []
    declaration_index = 0
    for entry in leaves:
        suite_id = suite_id_for_pytest_arguments(entry.pytest_arguments)
        gate_id = _gate_id_for_suite(suite_id, branch)
        if gate_id in gate_ids:
            continue
        gate_ids.append(gate_id)
        gates.append(
            QualificationGateDefinition(
                gate_id=gate_id,
                requires=(suite_id,),
                declaration_index=declaration_index,
            ),
        )
        declaration_index += 1
    aggregate_id = f"{branch}.aggregate"
    gates.append(
        QualificationGateDefinition(
            gate_id=aggregate_id,
            requires=tuple(gate_ids),
            declaration_index=declaration_index,
        ),
    )
    declaration_index += 1
    gates.append(
        QualificationGateDefinition(
            gate_id=root_gate_id,
            requires=(aggregate_id,),
            declaration_index=declaration_index,
        ),
    )
    profile = QualificationProfile(profile_id=profile_id, root_gate_ids=(root_gate_id,))
    graph = QualificationGraphDefinition(
        run_manifest=manifest,
        gates=tuple(gates),
        profiles=(profile,),
    )
    plan = compile_qualification_execution_plan(manifest, tuple(gates), profile)
    suite_by_id = MappingProxyType({suite.suite_id: suite for suite in suites})
    return CompiledCatalogProfile(
        graph=graph,
        plan=plan,
        suite_by_id=suite_by_id,
        shared_suite_ids=(),
    )


def _direct_suites_for_mandatory(
    top_mandatory: FrozenPytestSuiteSource,
    direct_label_to_suite_id: Mapping[str, str],
    skip_direct_labels: frozenset[str],
) -> tuple[QualificationSuite, ...]:
    suites: list[QualificationSuite] = []
    for label, targets in top_mandatory:
        if label in skip_direct_labels:
            continue
        suite_id = direct_label_to_suite_id[label]
        suites.append(
            QualificationSuite(
                suite_id=suite_id,
                pytest_arguments=normalize_pytest_arguments(targets),
                exclusive_resource_id=_exclusive_resource_for_suite_id(suite_id),
            ),
        )
    return tuple(suites)


def _npsc5f_r1_style_profile(
    *,
    profile_id: str,
    root_gate_id: str,
    top_mandatory: FrozenPytestSuiteSource,
    direct_label_to_suite_id: Mapping[str, str],
    npsc5e_expansion_mandatory: FrozenPytestSuiteSource,
    skip_direct_labels: frozenset[str],
    direct_branch: str,
    npsc5e_branch: str,
    shared_suite_ids: tuple[str, ...] = (),
) -> CompiledCatalogProfile:
    from testing_support.execution_qualification.compiler import (
        merge_qualification_suites,
    )

    direct_suites = _direct_suites_for_mandatory(
        top_mandatory,
        direct_label_to_suite_id,
        skip_direct_labels,
    )
    npsc5e_leaves = unique_required_leaf_targets(npsc5e_expansion_mandatory)
    npsc5e_suites = _build_suites_from_leaves(npsc5e_leaves)
    suites = merge_qualification_suites(direct_suites, npsc5e_suites)
    manifest = QualificationRunManifest(suites=suites)
    gates: list[QualificationGateDefinition] = []
    declaration_index = 0

    direct_gate_ids: list[str] = []
    for label, _targets in top_mandatory:
        if label in skip_direct_labels:
            continue
        suite_id = direct_label_to_suite_id[label]
        gate_id = _gate_id_for_suite(suite_id, direct_branch)
        direct_gate_ids.append(gate_id)
        gates.append(
            QualificationGateDefinition(
                gate_id=gate_id,
                requires=(suite_id,),
                declaration_index=declaration_index,
            ),
        )
        declaration_index += 1

    npsc5e_gate_ids: list[str] = []
    for entry in npsc5e_leaves:
        suite_id = suite_id_for_pytest_arguments(entry.pytest_arguments)
        gate_id = _gate_id_for_suite(suite_id, npsc5e_branch)
        if gate_id in npsc5e_gate_ids:
            continue
        npsc5e_gate_ids.append(gate_id)
        gates.append(
            QualificationGateDefinition(
                gate_id=gate_id,
                requires=(suite_id,),
                declaration_index=declaration_index,
            ),
        )
        declaration_index += 1

    direct_aggregate_id = f"{direct_branch}.aggregate"
    npsc5e_aggregate_id = f"{npsc5e_branch}.aggregate"
    gates.append(
        QualificationGateDefinition(
            gate_id=direct_aggregate_id,
            requires=tuple(direct_gate_ids),
            declaration_index=declaration_index,
        ),
    )
    declaration_index += 1
    gates.append(
        QualificationGateDefinition(
            gate_id=npsc5e_aggregate_id,
            requires=tuple(npsc5e_gate_ids),
            declaration_index=declaration_index,
        ),
    )
    declaration_index += 1
    gates.append(
        QualificationGateDefinition(
            gate_id=root_gate_id,
            requires=(direct_aggregate_id, npsc5e_aggregate_id),
            declaration_index=declaration_index,
        ),
    )

    profile = QualificationProfile(profile_id=profile_id, root_gate_ids=(root_gate_id,))
    graph = QualificationGraphDefinition(
        run_manifest=manifest,
        gates=tuple(gates),
        profiles=(profile,),
    )
    plan = compile_qualification_execution_plan(manifest, tuple(gates), profile)
    suite_by_id = MappingProxyType({suite.suite_id: suite for suite in suites})
    return CompiledCatalogProfile(
        graph=graph,
        plan=plan,
        suite_by_id=suite_by_id,
        shared_suite_ids=shared_suite_ids,
    )


def build_npsc5f_r1_profile() -> CompiledCatalogProfile:
    from testing_support.execution_qualification.catalog.labels import (
        NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID,
        SHARED_DG001_SUITE_ID,
        SHARED_NPSC5D_SUITE_ID,
    )

    return _npsc5f_r1_style_profile(
        profile_id=NPSC5F_R1_PROFILE_ID,
        root_gate_id=NPSC5F_R1_ROOT_GATE_ID,
        top_mandatory=NPSC5F_R1_FINAL_MANDATORY,
        direct_label_to_suite_id=NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID,
        npsc5e_expansion_mandatory=NPSC5E_FINAL_MANDATORY,
        skip_direct_labels=frozenset({"NPSC-5E Final"}),
        direct_branch="npsc5f-r1.direct",
        npsc5e_branch="npsc5e-r3.expanded",
        shared_suite_ids=(SHARED_DG001_SUITE_ID, SHARED_NPSC5D_SUITE_ID),
    )


def build_npsc5e_r3_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5E_R3_PROFILE_ID,
        root_gate_id="npsc5e-r3.final",
        mandatory=NPSC5E_R3_FINAL_MANDATORY,
        branch="npsc5e-r3",
    )


def build_npsc5e_r2_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5E_R2_PROFILE_ID,
        root_gate_id="npsc5e-r2.final",
        mandatory=NPSC5E_R2_FINAL_MANDATORY,
        branch="npsc5e-r2",
    )


def build_npsc5e_r1_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5E_R1_PROFILE_ID,
        root_gate_id="npsc5e-r1.final",
        mandatory=NPSC5E_R1_FINAL_MANDATORY,
        branch="npsc5e-r1",
    )


def build_npsc5e_final_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5E_FINAL_PROFILE_ID,
        root_gate_id="npsc5e.final",
        mandatory=NPSC5E_FINAL_MANDATORY,
        branch="npsc5e-final",
    )


def build_npsc5f_r2_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5F_R2_PROFILE_ID,
        root_gate_id="npsc5f-r2.final",
        mandatory=NPSC5F_R2_FINAL_MANDATORY,
        branch="npsc5f-r2",
    )


def build_npsc5f_r3_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5F_R3_PROFILE_ID,
        root_gate_id="npsc5f-r3.final",
        mandatory=NPSC5F_R3_FINAL_MANDATORY,
        branch="npsc5f-r3",
    )


def build_npsc5f_r4_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5F_R4_PROFILE_ID,
        root_gate_id="npsc5f-r4.final",
        mandatory=NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
        branch="npsc5f-r4",
    )


def build_npsc5f_final_profile() -> CompiledCatalogProfile:
    return _flat_profile(
        profile_id=NPSC5F_FINAL_PROFILE_ID,
        root_gate_id="npsc5f.final",
        mandatory=NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
        branch="npsc5f-final",
    )


QualificationProfileBuilder = Callable[[], CompiledCatalogProfile]

_PROFILE_BUILDER_MAP: dict[str, QualificationProfileBuilder] = {
    NPSC5F_R1_PROFILE_ID: build_npsc5f_r1_profile,
    NPSC5E_R3_PROFILE_ID: build_npsc5e_r3_profile,
    NPSC5E_R2_PROFILE_ID: build_npsc5e_r2_profile,
    NPSC5E_R1_PROFILE_ID: build_npsc5e_r1_profile,
    NPSC5E_FINAL_PROFILE_ID: build_npsc5e_final_profile,
    NPSC5F_R2_PROFILE_ID: build_npsc5f_r2_profile,
    NPSC5F_R3_PROFILE_ID: build_npsc5f_r3_profile,
    NPSC5F_R4_PROFILE_ID: build_npsc5f_r4_profile,
    NPSC5F_FINAL_PROFILE_ID: build_npsc5f_final_profile,
}

PROFILE_BUILDERS: Mapping[str, QualificationProfileBuilder] = MappingProxyType(
    _PROFILE_BUILDER_MAP,
)
