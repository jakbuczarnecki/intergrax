# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import uuid

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.contracts.enums import ImpactStatus, SelectionStatus
from testing_support.change_safety.contracts.impact import ImpactAssessmentResult
from testing_support.change_safety.contracts.selection import SelectedTestNode, SelectionDecision, SelectionReason


def select_tests_for_impact(
    impact: ImpactAssessmentResult,
    *,
    catalog_node_ids: frozenset[str],
    mandatory_node_ids: frozenset[str],
) -> SelectionDecision:
    if impact.status is ImpactStatus.INTEGRITY_FAILURE:
        return SelectionDecision(
            decision_id=str(uuid.uuid4()),
            status=SelectionStatus.REFUSED_INTEGRITY,
            graph_digest=impact.graph_digest,
            inventory_digest=impact.inventory_digest,
            mandatory_manifest_digest=stable_digest_hex(sorted(mandatory_node_ids)),
            ruleset_version=impact.ruleset_version,
        )
    if impact.status is ImpactStatus.REFUSED:
        return SelectionDecision(
            decision_id=str(uuid.uuid4()),
            status=SelectionStatus.CONSERVATIVE_FULL,
            mandatory=tuple(
                SelectedTestNode(node_id=n, reason_rule_id="R7-CONSERVATIVE") for n in sorted(mandatory_node_ids)
            ),
            graph_digest=impact.graph_digest,
            inventory_digest=impact.inventory_digest,
            suite_collection_digest=stable_digest_hex(sorted(catalog_node_ids)),
            mandatory_manifest_digest=stable_digest_hex(sorted(mandatory_node_ids)),
            ruleset_version=impact.ruleset_version,
        )
    selected_ids: set[str] = set(mandatory_node_ids)
    reasons: list[SelectionReason] = []
    for node in impact.affected:
        if node.kind == "CONSERVATIVE_UNKNOWN":
            return SelectionDecision(
                decision_id=str(uuid.uuid4()),
                status=SelectionStatus.CONSERVATIVE_FULL,
                mandatory=tuple(
                    SelectedTestNode(node_id=n, reason_rule_id="R7-CONSERVATIVE") for n in sorted(mandatory_node_ids)
                ),
                reasons=(SelectionReason(node_id=node.node_id, rule_id="R7-CONSERVATIVE"),),
                graph_digest=impact.graph_digest,
                inventory_digest=impact.inventory_digest,
                suite_collection_digest=stable_digest_hex(sorted(catalog_node_ids)),
                mandatory_manifest_digest=stable_digest_hex(sorted(mandatory_node_ids)),
                ruleset_version=impact.ruleset_version,
            )
    qual_x_smoke = frozenset(n for n in catalog_node_ids if n.startswith("tests/qualification/qual_x/"))
    selected_ids.update(qual_x_smoke)
    if not selected_ids:
        return SelectionDecision(
            decision_id=str(uuid.uuid4()),
            status=SelectionStatus.REFUSED_INTEGRITY,
            graph_digest=impact.graph_digest,
            inventory_digest=impact.inventory_digest,
            ruleset_version=impact.ruleset_version,
        )
    return SelectionDecision(
        decision_id=str(uuid.uuid4()),
        status=SelectionStatus.SELECTED,
        selected=tuple(
            SelectedTestNode(node_id=n, reason_rule_id="R1-SURFACE" if n not in mandatory_node_ids else "R2-MANDATORY")
            for n in sorted(selected_ids)
        ),
        mandatory=tuple(SelectedTestNode(node_id=n, reason_rule_id="R2-MANDATORY") for n in sorted(mandatory_node_ids)),
        reasons=tuple(reasons),
        graph_digest=impact.graph_digest,
        inventory_digest=impact.inventory_digest,
        suite_collection_digest=stable_digest_hex(sorted(catalog_node_ids)),
        mandatory_manifest_digest=stable_digest_hex(sorted(mandatory_node_ids)),
        ruleset_version=impact.ruleset_version,
    )
