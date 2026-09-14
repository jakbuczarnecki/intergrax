# © Artur Czarnecki. All rights reserved.

"""Canonical qualification DAG profile for NPSC-5F/R1 Final (representative migration)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.execution_qualification.compiler import (
    compile_qualification_execution_plan,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunManifest,
    QualificationSuite,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
    QualificationGateDefinition,
    QualificationGraphDefinition,
    QualificationProfile,
)
from testing_support.npsc5f_r1_legacy_targets import (
    LegacyRequiredTarget,
    expand_legacy_mandatory_subprocesses,
    legacy_r1_final_required_leaf_targets,
    normalize_pytest_arguments,
    normalize_required_target_set,
)
from tests.unit.runtime.architecture import (
    test_npsc5e_final_recovery_plane_qualification_and_freeze as npsc5e_final,
)
from tests.unit.runtime.architecture import (
    test_npsc5e_r2_final_checkpoint_durable_resume_qualification as npsc5e_r2_final,
)
from tests.unit.runtime.architecture import (
    test_npsc5e_r3_final_child_fanout_partial_recovery_qualification as npsc5e_r3_final,
)
from tests.unit.runtime.architecture import (
    test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity as npsc5f_r1_final,
)
from tests.unit.runtime.architecture.npsc5e_r3_final_execution_qualification import (
    NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID,
    NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID,
)

NPSC5F_R1_PROFILE_ID = "npsc5f-r1-final"
NPSC5F_R1_ROOT_GATE_ID = "npsc5f-r1.final"

_SHARED_DG001_SUITE_ID = "dg001-lineage"
_SHARED_NPSC5D_SUITE_ID = "npsc5d-final"
_RUNTIME_EVENTS_SUITE_ID = "runtime-events"
_RUNTIME_OBSERVABILITY_SUITE_ID = "runtime-observability"

_NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R1 implementation gate": "npsc5f-r1.implementation-gate",
    "NPSC-5F P0 gate": "npsc5f-r1.p0-gate",
    "Runtime events suites": _RUNTIME_EVENTS_SUITE_ID,
    "Runtime observability suites": _RUNTIME_OBSERVABILITY_SUITE_ID,
    "DG_001": _SHARED_DG001_SUITE_ID,
    "NPSC-5D Final": _SHARED_NPSC5D_SUITE_ID,
}

_NPSC5E_R2_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R1 Final": "npsc5e-r2.mandatory.r1-final",
    "R2 Original": "npsc5e-r2.mandatory.r2-original",
    "R2-H1": "npsc5e-r2.mandatory.r2-h1",
    "R2-H2": "npsc5e-r2.mandatory.r2-h2",
    "R2-H2-Q1": "npsc5e-r2.mandatory.r2-h2-q1",
    "P0A": "npsc5e-r2.mandatory.p0a",
    "DG_001": _SHARED_DG001_SUITE_ID,
    "NPSC-5D Final": _SHARED_NPSC5D_SUITE_ID,
    "HITL R3": "npsc5e-r2.mandatory.hitl-r3",
    "NPSC-5A": "npsc5e-r2.mandatory.npsc-5a",
    "NPSC-5B": "npsc5e-r2.mandatory.npsc-5b",
    "NPSC-5C": "npsc5e-r2.mandatory.npsc-5c",
    "Attempt lifecycle": "npsc5e-r2.mandatory.attempt-lifecycle",
    "Child execution": "npsc5e-r2.mandatory.child-execution",
    "Terminal": "npsc5e-r2.mandatory.terminal",
    "Cancellation": "npsc5e-r2.mandatory.cancellation",
    "Checkpoint store": "npsc5e-r2.mandatory.checkpoint-store",
    "Long-running": "npsc5e-r2.mandatory.long-running",
}


def _register_label_map(
    mapping: dict[str, str],
    source: tuple[tuple[str, list[str]], ...],
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
                    raise ValueError(f"conflicting suite_id for pytest args: {key!r}")
                continue
            registry[key] = suite_id


def _build_pytest_to_suite_id() -> dict[tuple[str, ...], str]:
    registry: dict[tuple[str, ...], str] = {}
    r3_map = dict(NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID)
    r3_map["DG_001"] = _SHARED_DG001_SUITE_ID
    r3_map["NPSC-5D Final"] = _SHARED_NPSC5D_SUITE_ID
    _register_label_map(r3_map, npsc5e_r3_final._MANDATORY_SUITES, registry)
    _register_label_map(
        _NPSC5E_R2_LABEL_TO_SUITE_ID,
        npsc5e_r2_final._MANDATORY_SUITES,
        registry,
        skip_existing=True,
    )
    _register_label_map(
        _NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID,
        npsc5f_r1_final._MANDATORY_SUITES,
        registry,
        skip_existing=True,
    )
    return registry


_PYTEST_TO_SUITE_ID = _build_pytest_to_suite_id()


@dataclass(frozen=True, slots=True)
class Npsc5fR1QualificationGraph:
    graph: QualificationGraphDefinition
    plan: QualificationExecutionPlan
    suite_by_id: dict[str, QualificationSuite]
    shared_suite_ids: tuple[str, ...]


def suite_id_for_pytest_arguments(pytest_arguments: tuple[str, ...]) -> str:
    key = normalize_pytest_arguments(pytest_arguments)
    if key not in _PYTEST_TO_SUITE_ID:
        raise ValueError(f"no suite_id for pytest arguments: {key!r}")
    return _PYTEST_TO_SUITE_ID[key]


def _exclusive_resource_for_suite_id(suite_id: str) -> str | None:
    if suite_id == NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID["R3 implementation gate"]:
        return NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID
    return None


def _unique_leaves_in_order(
    entries: tuple[LegacyRequiredTarget, ...],
) -> tuple[LegacyRequiredTarget, ...]:
    seen: set[tuple[str, ...]] = set()
    unique: list[LegacyRequiredTarget] = []
    for entry in entries:
        if entry.pytest_arguments in seen:
            continue
        seen.add(entry.pytest_arguments)
        unique.append(entry)
    return tuple(unique)


def _npsc5e_expanded_leaves() -> tuple[LegacyRequiredTarget, ...]:
    return _unique_leaves_in_order(
        expand_legacy_mandatory_subprocesses(npsc5e_final._MANDATORY_SUITES),
    )


def _build_suites(
    leaves: tuple[LegacyRequiredTarget, ...],
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


def build_npsc5f_r1_qualification_graph() -> Npsc5fR1QualificationGraph:
    all_leaves = legacy_r1_final_required_leaf_targets()
    suites = _build_suites(all_leaves)
    suite_by_id = {suite.suite_id: suite for suite in suites}
    manifest = QualificationRunManifest(suites=suites)

    direct_branch = "npsc5f-r1.direct"
    npsc5e_branch = "npsc5e-r3.expanded"
    gates: list[QualificationGateDefinition] = []
    declaration_index = 0

    direct_gate_ids: list[str] = []
    for label, _targets in npsc5f_r1_final._MANDATORY_SUITES:
        if label == "NPSC-5E Final":
            continue
        suite_id = _NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID[label]
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
    for entry in _npsc5e_expanded_leaves():
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
            gate_id=NPSC5F_R1_ROOT_GATE_ID,
            requires=(direct_aggregate_id, npsc5e_aggregate_id),
            declaration_index=declaration_index,
        ),
    )

    profile = QualificationProfile(
        profile_id=NPSC5F_R1_PROFILE_ID,
        root_gate_ids=(NPSC5F_R1_ROOT_GATE_ID,),
    )
    graph = QualificationGraphDefinition(
        run_manifest=manifest,
        gates=tuple(gates),
        profiles=(profile,),
    )
    plan = compile_qualification_execution_plan(manifest, tuple(gates), profile)
    return Npsc5fR1QualificationGraph(
        graph=graph,
        plan=plan,
        suite_by_id=suite_by_id,
        shared_suite_ids=(_SHARED_DG001_SUITE_ID, _SHARED_NPSC5D_SUITE_ID),
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
