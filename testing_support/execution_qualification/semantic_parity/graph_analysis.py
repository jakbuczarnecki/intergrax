# © Artur Czarnecki. All rights reserved.

"""Pure graph helpers for qualification plan semantic certification."""

from __future__ import annotations

from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
    QualificationExecutionNode,
    QualificationNodeKind,
)


def canonical_leaf_pytest_argument_set(
    plan: QualificationExecutionPlan,
) -> frozenset[tuple[str, ...]]:
    args: set[tuple[str, ...]] = set()
    for node in plan.ordered_nodes:
        if node.kind is QualificationNodeKind.LEAF_SUITE and node.suite is not None:
            args.add(node.suite.pytest_arguments)
    return frozenset(args)


def _node_by_id(
    plan: QualificationExecutionPlan,
) -> dict[str, QualificationExecutionNode]:
    return {node.node_id: node for node in plan.ordered_nodes}


def transitive_dependency_closure(
    plan: QualificationExecutionPlan,
    start_node_id: str,
) -> frozenset[str]:
    nodes = _node_by_id(plan)
    visited: set[str] = set()
    stack = [start_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = nodes.get(node_id)
        if node is None:
            continue
        for dep in node.dependencies:
            stack.append(dep)
    return frozenset(visited)


def nodes_reachable_from_roots(plan: QualificationExecutionPlan) -> frozenset[str]:
    reachable: set[str] = set()
    for root_id in plan.root_gate_ids:
        reachable |= set(transitive_dependency_closure(plan, root_id))
    return frozenset(reachable)


def root_ids_transitively_requiring_leaf(
    plan: QualificationExecutionPlan,
    leaf_suite_id: str,
) -> tuple[str, ...]:
    affected: list[str] = []
    for root_id in plan.root_gate_ids:
        closure = transitive_dependency_closure(plan, root_id)
        if leaf_suite_id in closure:
            affected.append(root_id)
    return tuple(affected)


def reachable_non_leaf_gate_count(plan: QualificationExecutionPlan) -> int:
    reachable = nodes_reachable_from_roots(plan)
    count = 0
    for node in plan.ordered_nodes:
        if node.node_id not in reachable:
            continue
        if node.kind is not QualificationNodeKind.LEAF_SUITE:
            count += 1
    return count
