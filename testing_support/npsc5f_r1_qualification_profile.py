# © Artur Czarnecki. All rights reserved.

"""Canonical qualification DAG profile for NPSC-5F/R1 Final (catalog-backed)."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from testing_support.execution_qualification.catalog.labels import (
    SHARED_DG001_SUITE_ID,
    SHARED_NPSC5D_SUITE_ID,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5F_R1_PROFILE_ID,
    NPSC5F_R1_ROOT_GATE_ID,
    build_npsc5f_r1_profile,
)
from testing_support.execution_qualification.catalog.suite_registry import (
    suite_id_for_pytest_arguments,
)
from testing_support.execution_qualification.contracts import QualificationSuite
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
    QualificationGraphDefinition,
)
from testing_support.npsc5f_r1_legacy_targets import (
    legacy_r1_final_required_leaf_targets,
    normalize_pytest_arguments,
    normalize_required_target_set,
)

_SHARED_DG001_SUITE_ID = SHARED_DG001_SUITE_ID
_SHARED_NPSC5D_SUITE_ID = SHARED_NPSC5D_SUITE_ID


@dataclass(frozen=True, slots=True)
class Npsc5fR1QualificationGraph:
    graph: QualificationGraphDefinition
    plan: QualificationExecutionPlan
    suite_by_id: MappingProxyType[str, QualificationSuite]
    shared_suite_ids: tuple[str, ...]


def build_npsc5f_r1_qualification_graph() -> Npsc5fR1QualificationGraph:
    compiled = build_npsc5f_r1_profile()
    return Npsc5fR1QualificationGraph(
        graph=compiled.graph,
        plan=compiled.plan,
        suite_by_id=MappingProxyType(dict(compiled.suite_by_id)),
        shared_suite_ids=compiled.shared_suite_ids,
    )


def dag_required_target_set(
    plan: QualificationExecutionPlan,
) -> frozenset[tuple[str, ...]]:
    args: set[tuple[str, ...]] = set()
    for node in plan.ordered_nodes:
        if node.suite is not None:
            args.add(node.suite.pytest_arguments)
    return frozenset(args)


def legacy_and_dag_required_target_sets_equal() -> bool:
    built = build_npsc5f_r1_qualification_graph()
    legacy_set = normalize_required_target_set(legacy_r1_final_required_leaf_targets())
    dag_set = dag_required_target_set(built.plan)
    return legacy_set == dag_set


__all__ = [
    "NPSC5F_R1_PROFILE_ID",
    "NPSC5F_R1_ROOT_GATE_ID",
    "Npsc5fR1QualificationGraph",
    "SHARED_DG001_SUITE_ID",
    "_SHARED_DG001_SUITE_ID",
    "_SHARED_NPSC5D_SUITE_ID",
    "build_npsc5f_r1_qualification_graph",
    "dag_required_target_set",
    "legacy_and_dag_required_target_sets_equal",
    "normalize_pytest_arguments",
    "suite_id_for_pytest_arguments",
]
