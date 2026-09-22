# © Artur Czarnecki. All rights reserved.

"""Shared helpers for GR-12 control-plane surface qualification tests."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from tests.qualification.governance.gr12.a3_path_qualifications import (
    GR12_A3_AD_PATH_TO_MUTATION_TYPE,
    GR12_A3_COMPOSITION_PROOF_SSOT,
    GR12_A3_SHARED_MECHANISM_BY_ID,
    Gr12A3PathProofBundle,
    Gr12CompositionDomain,
    Gr12ProofInvariant,
    Gr12QualificationPathKind,
    Gr12SharedQualificationMechanism,
    gr12_a3_composition_proof_nodes,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]

_MUTATION_INVARIANTS = frozenset(
    {
        Gr12ProofInvariant.ALLOW,
        Gr12ProofInvariant.DENY,
        Gr12ProofInvariant.TENANT,
        Gr12ProofInvariant.STALE,
        Gr12ProofInvariant.HITL,
        Gr12ProofInvariant.EVIDENCE,
        Gr12ProofInvariant.PLUGINABILITY,
    }
)

_COMPOSITION_INVARIANTS = frozenset(
    {
        Gr12ProofInvariant.AUTHORITY_REQUIRED,
        Gr12ProofInvariant.MISSING_AUTHORITY_FAIL_CLOSED,
        Gr12ProofInvariant.CANONICAL_CONSUMER,
        Gr12ProofInvariant.NO_DUPLICATE_AUTHORITY,
        Gr12ProofInvariant.EXTERNAL_EVALUATOR,
    }
)


def _function_names_in_test_module(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def assert_proof_nodes_registered(node_ids: tuple[str, ...]) -> None:
    missing: list[str] = []
    for node_id in node_ids:
        rel, name = node_id.split("::", 1)
        module_path = _REPO_ROOT / rel
        if not module_path.is_file():
            missing.append(node_id)
            continue
        if name not in _function_names_in_test_module(module_path):
            missing.append(node_id)
    assert missing == [], f"missing proof tests: {missing}"


def _invariants_for_nodes(
    nodes: tuple,
    *,
    path_id: str,
    shared_mechanism_id: str | None,
) -> set[Gr12ProofInvariant]:
    result: set[Gr12ProofInvariant] = set()
    for node in nodes:
        if node.direct_path_id == path_id:
            result.add(node.invariant)
        if shared_mechanism_id is not None and node.shared_mechanism_id == shared_mechanism_id:
            result.add(node.invariant)
    return result


def _assert_shared_mechanism(
    mechanism: Gr12SharedQualificationMechanism,
    *,
    path_id: str,
    nodes: tuple,
) -> None:
    assert path_id in mechanism.applicable_path_ids, path_id
    assert_proof_nodes_registered(mechanism.proof_tests)
    shared_invariants = _invariants_for_nodes(
        nodes,
        path_id=path_id,
        shared_mechanism_id=mechanism.mechanism_id,
    )
    if mechanism.hitl_enforcement_owner is not None:
        assert Gr12ProofInvariant.HITL in shared_invariants, path_id
        assert Gr12ProofInvariant.ALLOW in _invariants_for_nodes(
            nodes,
            path_id=path_id,
            shared_mechanism_id=None,
        ), f"{path_id} missing ALLOW wiring for shared HITL"
    if mechanism.stale_guard_owner is not None:
        assert Gr12ProofInvariant.STALE in shared_invariants, path_id


_COMPOSITION_PATH_DOMAIN: Final[dict[str, Gr12CompositionDomain]] = {
    "CP-HOST-BOUNDARY-OPTIONAL": Gr12CompositionDomain.HOST_TASK_CONTROL,
    "CP-ECP-BOUNDARY-OPTIONAL": Gr12CompositionDomain.ECP,
}


def assert_gr12_a3_composition_proof_structural_integrity(bundle: Gr12A3PathProofBundle) -> None:
    assert bundle.kind is Gr12QualificationPathKind.COMPOSITION_SURFACE, bundle.path_id
    assert bundle.proof_nodes, f"{bundle.path_id} composition must use explicit proof_nodes"
    nodes = bundle.proof_nodes
    assert nodes == gr12_a3_composition_proof_nodes(bundle.path_id), bundle.path_id
    ssot = GR12_A3_COMPOSITION_PROOF_SSOT[bundle.path_id]
    invariants_seen: set[Gr12ProofInvariant] = set()
    for node in nodes:
        assert node.direct_path_id == bundle.path_id, node.test_id
        assert node.shared_mechanism_id is None, node.test_id
        assert node.invariant in _COMPOSITION_INVARIANTS, node.test_id
        assert node.invariant not in invariants_seen, f"duplicate invariant {node.invariant}"
        invariants_seen.add(node.invariant)
        expected_test_id, expected_domain = ssot[node.invariant]
        assert node.test_id == expected_test_id, (node.invariant, node.test_id)
        assert node.composition_domain == expected_domain, node.test_id
    missing = _COMPOSITION_INVARIANTS - invariants_seen
    assert not missing, f"{bundle.path_id} missing composition invariants: {sorted(missing)}"
    path_domain = _COMPOSITION_PATH_DOMAIN[bundle.path_id]
    for node in nodes:
        assert node.composition_domain == path_domain, (
            f"{node.test_id} domain {node.composition_domain} != path {path_domain}"
        )


def assert_gr12_a3_path_semantic_integrity(bundle: Gr12A3PathProofBundle) -> None:
    nodes = bundle.resolved_proof_nodes()
    if bundle.kind is Gr12QualificationPathKind.COMPOSITION_SURFACE:
        assert_gr12_a3_composition_proof_structural_integrity(bundle)
        present = {node.invariant for node in nodes if node.direct_path_id == bundle.path_id}
        missing = _COMPOSITION_INVARIANTS - present
        assert not missing, f"{bundle.path_id} missing composition invariants: {sorted(missing)}"
        assert not (_MUTATION_INVARIANTS & present), bundle.path_id
        return

    present = _invariants_for_nodes(nodes, path_id=bundle.path_id, shared_mechanism_id=None)
    for node in nodes:
        if node.shared_mechanism_id is not None:
            mechanism = GR12_A3_SHARED_MECHANISM_BY_ID[node.shared_mechanism_id]
            _assert_shared_mechanism(mechanism, path_id=bundle.path_id, nodes=nodes)
            if node.invariant is Gr12ProofInvariant.STALE:
                present.add(Gr12ProofInvariant.STALE)
            if node.invariant is Gr12ProofInvariant.HITL:
                present.add(Gr12ProofInvariant.HITL)

    missing = _MUTATION_INVARIANTS - present
    assert not missing, f"{bundle.path_id} missing mutation invariants: {sorted(missing)}"

    if bundle.path_id in GR12_A3_AD_PATH_TO_MUTATION_TYPE:
        assert bundle.kind is Gr12QualificationPathKind.MUTATION_SURFACE, bundle.path_id
