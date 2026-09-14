# © Artur Czarnecki. All rights reserved.

"""Deterministic qualification DAG compiler: definition → validation → execution plan."""

from __future__ import annotations

from collections.abc import Sequence

from testing_support.execution_qualification.contracts import (
    QualificationRunManifest,
    QualificationSuite,
)
from testing_support.execution_qualification.graph_contracts import (
    ConflictingQualificationSuiteDefinitionError,
    MissingQualificationDependencyError,
    QualificationDependencyCycleError,
    QualificationExecutionNode,
    QualificationExecutionPlan,
    QualificationGateDefinition,
    QualificationManifestConflictError,
    QualificationNodeKind,
    QualificationProfile,
    QualificationProfileError,
)


def merge_qualification_suites(
    *suite_groups: Sequence[QualificationSuite],
) -> tuple[QualificationSuite, ...]:
    """Build a deduplicated suite tuple; fail closed on conflicting definitions."""

    registry: dict[str, QualificationSuite] = {}
    order: list[str] = []
    for group in suite_groups:
        for suite in group:
            existing = registry.get(suite.suite_id)
            if existing is None:
                registry[suite.suite_id] = suite
                order.append(suite.suite_id)
                continue
            if existing != suite:
                raise ConflictingQualificationSuiteDefinitionError(
                    f"conflicting suite definition for {suite.suite_id!r}"
                )
    return tuple(registry[sid] for sid in order)


def _suite_declaration_index(manifest: QualificationRunManifest) -> dict[str, int]:
    return {suite.suite_id: index for index, suite in enumerate(manifest.suites)}


def _build_gate_map(
    gates: tuple[QualificationGateDefinition, ...],
) -> dict[str, QualificationGateDefinition]:
    gate_map: dict[str, QualificationGateDefinition] = {}
    for gate in gates:
        if gate.gate_id in gate_map:
            raise QualificationManifestConflictError(
                f"duplicate gate_id: {gate.gate_id}"
            )
        gate_map[gate.gate_id] = gate
    return gate_map


def _build_suite_map(
    manifest: QualificationRunManifest,
    gate_map: dict[str, QualificationGateDefinition],
) -> dict[str, QualificationSuite]:
    suite_map: dict[str, QualificationSuite] = {}
    for suite in manifest.suites:
        if suite.suite_id in gate_map:
            raise QualificationManifestConflictError(
                f"suite_id collides with gate_id: {suite.suite_id}"
            )
        if suite.suite_id in suite_map:
            raise QualificationManifestConflictError(
                f"duplicate suite_id: {suite.suite_id}"
            )
        suite_map[suite.suite_id] = suite
    return suite_map


def _resolve_node_kind(
    node_id: str,
    gate_map: dict[str, QualificationGateDefinition],
    suite_map: dict[str, QualificationSuite],
) -> QualificationNodeKind:
    if node_id in suite_map:
        return QualificationNodeKind.LEAF_SUITE
    gate = gate_map[node_id]
    return gate.gate_kind


def _collect_reachable(
    root_gate_ids: tuple[str, ...],
    gate_map: dict[str, QualificationGateDefinition],
    suite_map: dict[str, QualificationSuite],
) -> tuple[set[str], set[str]]:
    reachable_gates: set[str] = set()
    reachable_suites: set[str] = set()

    def visit_gate(gate_id: str, stack: tuple[str, ...]) -> None:
        if gate_id in reachable_gates:
            return
        if gate_id in stack:
            idx = stack.index(gate_id)
            raise QualificationDependencyCycleError(stack[idx:] + (gate_id,))
        gate = gate_map[gate_id]
        next_stack = stack + (gate_id,)
        for dep in gate.requires:
            if dep == gate_id:
                raise QualificationDependencyCycleError((gate_id, gate_id))
            if dep in suite_map:
                reachable_suites.add(dep)
            elif dep in gate_map:
                visit_gate(dep, next_stack)
            else:
                raise MissingQualificationDependencyError(
                    f"missing dependency {dep!r} required by gate {gate_id!r}"
                )
        reachable_gates.add(gate_id)

    for root_id in root_gate_ids:
        if root_id not in gate_map:
            raise QualificationProfileError(f"unknown root gate: {root_id!r}")
        visit_gate(root_id, ())

    return reachable_gates, reachable_suites


def _topological_order(
    reachable_gates: set[str],
    reachable_suites: set[str],
    gate_map: dict[str, QualificationGateDefinition],
    suite_decl_index: dict[str, int],
) -> list[str]:
    nodes = reachable_gates | reachable_suites
    in_degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    successors: dict[str, list[str]] = {node_id: [] for node_id in nodes}

    for gate_id in reachable_gates:
        gate = gate_map[gate_id]
        for dep in gate.requires:
            if dep not in nodes:
                continue
            in_degree[gate_id] += 1
            successors[dep].append(gate_id)

    def sort_key(node_id: str) -> tuple[int, str]:
        if node_id in gate_map:
            return (gate_map[node_id].declaration_index, node_id)
        return (suite_decl_index[node_id], node_id)

    ready = sorted(
        (node_id for node_id in nodes if in_degree[node_id] == 0),
        key=sort_key,
    )
    ordered: list[str] = []
    layer = 0
    layer_by_node: dict[str, int] = {}

    while ready:
        for node_id in ready:
            layer_by_node[node_id] = layer
            ordered.append(node_id)
        next_ready: list[str] = []
        for node_id in ready:
            for succ in successors[node_id]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    next_ready.append(succ)
        layer += 1
        ready = sorted(next_ready, key=sort_key)

    if len(ordered) != len(nodes):
        raise QualificationDependencyCycleError(("cycle",))

    ordered.sort(key=lambda node_id: (layer_by_node[node_id], sort_key(node_id)))
    return ordered


def _dependency_edges(
    reachable_gates: set[str],
    gate_map: dict[str, QualificationGateDefinition],
    suite_map: dict[str, QualificationSuite],
) -> tuple[tuple[str, str], ...]:
    edges: list[tuple[str, str]] = []
    for gate_id in sorted(reachable_gates):
        for dep in gate_map[gate_id].requires:
            if dep in suite_map or dep in gate_map:
                edges.append((dep, gate_id))
    return tuple(edges)


def compile_qualification_execution_plan(
    manifest: QualificationRunManifest,
    gates: tuple[QualificationGateDefinition, ...],
    profile: QualificationProfile,
) -> QualificationExecutionPlan:
    """Pure compiler: manifest + gates + profile → immutable execution plan."""

    if not profile.root_gate_ids:
        raise QualificationProfileError("profile must declare at least one root gate")

    gate_map = _build_gate_map(gates)
    suite_map = _build_suite_map(manifest, gate_map)
    suite_decl_index = _suite_declaration_index(manifest)

    reachable_gates, reachable_suites = _collect_reachable(
        profile.root_gate_ids, gate_map, suite_map
    )
    ordered_ids = _topological_order(
        reachable_gates, reachable_suites, gate_map, suite_decl_index
    )

    execution_nodes: list[QualificationExecutionNode] = []
    leaf_suite_ids: list[str] = []

    for node_id in ordered_ids:
        if node_id in suite_map:
            suite = suite_map[node_id]
            deps: tuple[str, ...] = ()
            execution_nodes.append(
                QualificationExecutionNode(
                    node_id=node_id,
                    kind=QualificationNodeKind.LEAF_SUITE,
                    dependencies=deps,
                    declaration_index=suite_decl_index[node_id],
                    suite=suite,
                    gate=None,
                )
            )
            leaf_suite_ids.append(node_id)
            continue

        gate = gate_map[node_id]
        execution_nodes.append(
            QualificationExecutionNode(
                node_id=node_id,
                kind=gate.gate_kind,
                dependencies=gate.requires,
                declaration_index=gate.declaration_index,
                suite=None,
                gate=gate,
            )
        )

    return QualificationExecutionPlan(
        profile_id=profile.profile_id,
        ordered_nodes=tuple(execution_nodes),
        leaf_suite_ids=tuple(leaf_suite_ids),
        root_gate_ids=profile.root_gate_ids,
        dependency_edges=_dependency_edges(reachable_gates, gate_map, suite_map),
    )


class QualificationGraphCompiler:
    """Stateless façade over :func:`compile_qualification_execution_plan`."""

    def compile(
        self,
        manifest: QualificationRunManifest,
        gates: tuple[QualificationGateDefinition, ...],
        profile: QualificationProfile,
    ) -> QualificationExecutionPlan:
        return compile_qualification_execution_plan(manifest, gates, profile)
