# © Artur Czarnecki. All rights reserved.

"""M-09H attributed cause identity — deterministic oracle @ TEST_EXECUTION_SHA (OD-EVOL-08-M1).

Semantic: the seeded path change is attributed to the **DIRECT** semantic-graph node for that
module. The cause identity is the stable digest of the typed pair ``(kind, node_id)`` using
``stable_digest_hex`` (canonical JSON serialization — see ``contracts/canonical_json.py``).

Normative: ``docs/project/maintainers/qualification/evol_x/EVOL_X_M09H_CAUSE_IDENTITY.md``.
"""

from __future__ import annotations

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.contracts.impact import AffectedNode, ImpactAssessmentResult
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import module_id_for_path
from testing_support.change_safety.graph.impact import assess_path_change_impact

M09_CAUSE_IDENTITY_SCHEMA = "evol_m09_cause_identity.v1"


def normalize_seed_path(changed_path: str) -> str:
    return changed_path.replace("\\", "/")


def direct_module_node_id_for_seed(changed_path: str) -> str:
    return module_id_for_path(normalize_seed_path(changed_path))


def cause_identity_digest_for_node(node: AffectedNode) -> str:
    """Single canonical cause identity digest for a graph node."""
    return stable_digest_hex([node.kind, node.node_id])


def direct_seed_cause_node(impact: ImpactAssessmentResult, changed_path: str) -> AffectedNode:
    """Return the DIRECT affected node for the seed module (fail-closed if ambiguous)."""
    target = direct_module_node_id_for_seed(changed_path)
    matches = [n for n in impact.affected if n.kind == "DIRECT" and n.node_id == target]
    if len(matches) != 1:
        raise ValueError(
            f"M09 seed {changed_path!r}: expected exactly one DIRECT node {target!r}, "
            f"found {len(matches)} (impact status={impact.status.value})"
        )
    return matches[0]


def assess_seed_path_impact(inventory: ClosedWorldInventory, changed_path: str) -> ImpactAssessmentResult:
    path = normalize_seed_path(changed_path)
    return assess_path_change_impact(inventory, changed_paths=frozenset({path}))


def expected_cause_identities_for_seed(inventory: ClosedWorldInventory, changed_path: str) -> tuple[str, ...]:
    """Oracle tuple for registry ``expected_cause_identities`` (length 1 for C1..C3 seeds)."""
    impact = assess_seed_path_impact(inventory, changed_path)
    node = direct_seed_cause_node(impact, changed_path)
    return (cause_identity_digest_for_node(node),)


def describe_attribution_evidence(impact: ImpactAssessmentResult, changed_path: str) -> dict[str, str]:
    """Human-facing facts from diagnostic output (no expected digest)."""
    node = direct_seed_cause_node(impact, changed_path)
    return {
        "schema_version": M09_CAUSE_IDENTITY_SCHEMA,
        "seed_changed_path": normalize_seed_path(changed_path),
        "attributed_node_kind": node.kind,
        "attributed_node_id": node.node_id,
        "identity_rule": "stable_digest_hex([kind, node_id])",
    }
