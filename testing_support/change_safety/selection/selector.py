# © Artur Czarnecki. All rights reserved.

"""W3-05: EVOL_CONSERVATIVE_FULL selector (T1 ∪ T2 ∪ T3 ∪ Adj \\ exclusions), fail-closed, deterministic decision id."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.contracts.enums import ImpactStatus, SelectionStatus
from testing_support.change_safety.contracts.impact import ImpactAssessmentResult
from testing_support.change_safety.contracts.selection import SelectedTestNode, SelectionDecision, SelectionReason
from testing_support.change_safety.selection.policy import (
    REASON_RULE_SCOPED,
    ConservativeExclusions,
    evol_conservative_full_set,
)

TOOL_VERSION = "evol-x-0.1.0"
_SELF_SURFACE_PREFIXES = ("module:testing_support.change_safety", "suite:tests/qualification/evol_x/")


@dataclass(frozen=True)
class SelectionInputs:
    impact: ImpactAssessmentResult | None
    catalog_node_ids: frozenset[str]
    mandatory_node_ids: frozenset[str]
    pop_a_node_ids: frozenset[str] = frozenset()
    evol_manifest_node_ids: frozenset[str] = frozenset()
    adjacent_node_ids: frozenset[str] = frozenset()
    exclusions: ConservativeExclusions | None = None
    collectable_node_ids: frozenset[str] | None = None
    required_suite_ids: frozenset[str] = frozenset()
    expected_head_sha: str | None = None
    expected_graph_digest: str | None = None
    expected_inventory_digest: str | None = None
    critical_omissions: int = 0
    analyzer: Callable[[], ImpactAssessmentResult] | None = field(default=None, compare=False)
    modules_with_unknown_deps: frozenset[str] = frozenset()


def _decision(
    status: SelectionStatus,
    inputs: SelectionInputs,
    impact: ImpactAssessmentResult | None,
    *,
    selected: dict[str, str],
    reasons: tuple[SelectionReason, ...] = (),
) -> SelectionDecision:
    mandatory = {n: "R2-MANDATORY" for n in sorted(inputs.mandatory_node_ids)}
    mandatory_digest = stable_digest_hex(sorted(inputs.mandatory_node_ids))
    suite_digest = stable_digest_hex(sorted(inputs.catalog_node_ids))
    graph_digest = impact.graph_digest if impact else ""
    inventory_digest = impact.inventory_digest if impact else ""
    ruleset = impact.ruleset_version if impact else ""
    identity = {
        "status": status.value,
        "selected": sorted(selected.items()),
        "reasons": [[r.node_id, r.rule_id, r.witness] for r in reasons],
        "graph_digest": graph_digest,
        "inventory_digest": inventory_digest,
        "suite_collection_digest": suite_digest,
        "mandatory_manifest_digest": mandatory_digest,
        "ruleset_version": ruleset,
        "tool_version": TOOL_VERSION,
    }
    return SelectionDecision(
        decision_id=stable_digest_hex(identity)[:32],
        status=status,
        selected=tuple(SelectedTestNode(node_id=n, reason_rule_id=r) for n, r in sorted(selected.items())),
        mandatory=tuple(SelectedTestNode(node_id=n, reason_rule_id=r) for n, r in mandatory.items()),
        reasons=reasons,
        graph_digest=graph_digest,
        inventory_digest=inventory_digest,
        suite_collection_digest=suite_digest,
        mandatory_manifest_digest=mandatory_digest,
        ruleset_version=ruleset,
        tool_version=TOOL_VERSION,
    )


def _refuse(
    inputs: SelectionInputs, impact: ImpactAssessmentResult | None, fault: str, witness: str = ""
) -> SelectionDecision:
    return _decision(
        SelectionStatus.REFUSED_INTEGRITY,
        inputs,
        impact,
        selected={},
        reasons=(SelectionReason(node_id="selection", rule_id=f"FAULT:{fault}", witness=witness),),
    )


def _conservative(
    inputs: SelectionInputs, impact: ImpactAssessmentResult | None, fault: str, witness: str, universe: frozenset[str]
) -> SelectionDecision:
    return _decision(
        SelectionStatus.CONSERVATIVE_FULL,
        inputs,
        impact,
        selected={n: REASON_RULE_SCOPED for n in universe},
        reasons=(
            SelectionReason(node_id="selection", rule_id=f"FAULT:{fault}", witness=witness),
            SelectionReason(node_id="selection", rule_id=REASON_RULE_SCOPED, witness="policy EVOL_CONSERVATIVE_FULL"),
        ),
    )


def _node_module(node_id: str) -> str:
    return node_id.split("::", 1)[0]


def select_tests(inputs: SelectionInputs) -> SelectionDecision:
    universe = evol_conservative_full_set(
        t1=inputs.mandatory_node_ids,
        t2=inputs.pop_a_node_ids,
        t3=inputs.evol_manifest_node_ids,
        adjacent=inputs.adjacent_node_ids,
        exclusions=inputs.exclusions,
    )
    impact = inputs.impact
    if inputs.analyzer is not None:
        try:
            impact = inputs.analyzer()
        except Exception as exc:  # noqa: BLE001 - fail-closed on any analyzer fault (fault class ANALYZER_EXCEPTION)
            return _conservative(inputs, None, "ANALYZER_EXCEPTION", type(exc).__name__, universe)
    if impact is None:
        return _refuse(inputs, None, "MISSING_METADATA", "impact missing")
    if not impact.ruleset_version or not inputs.mandatory_node_ids:
        return _refuse(inputs, impact, "MISSING_METADATA", "ruleset_version or mandatory manifest empty")
    if not inputs.catalog_node_ids:
        return _refuse(inputs, impact, "EMPTY_CATALOG")
    catalog_modules = {_node_module(n) for n in inputs.catalog_node_ids | inputs.evol_manifest_node_ids}
    missing_suites = sorted(inputs.required_suite_ids - catalog_modules)
    if missing_suites or not inputs.mandatory_node_ids <= inputs.catalog_node_ids:
        return _refuse(inputs, impact, "MISSING_SUITE_ID", ",".join(missing_suites[:5]) or "mandatory not in catalog")
    if inputs.expected_head_sha is not None and impact.source_commit_sha != inputs.expected_head_sha:
        return _refuse(inputs, impact, "STALE_SNAPSHOT", impact.source_commit_sha)
    if (inputs.expected_graph_digest is not None and impact.graph_digest != inputs.expected_graph_digest) or (
        inputs.expected_inventory_digest is not None and impact.inventory_digest != inputs.expected_inventory_digest
    ):
        return _refuse(inputs, impact, "DIGEST_MISMATCH")
    if impact.status is ImpactStatus.INTEGRITY_FAILURE:
        return _refuse(inputs, impact, "DIGEST_MISMATCH", "impact INTEGRITY_FAILURE")
    required = frozenset(
        n
        for n in (
            inputs.mandatory_node_ids | inputs.pop_a_node_ids | inputs.evol_manifest_node_ids | inputs.adjacent_node_ids
        )
        if not (inputs.exclusions and inputs.exclusions.excludes(n))
    )
    if inputs.collectable_node_ids is not None:
        uncollectable = sorted(required - inputs.collectable_node_ids)
        if uncollectable:
            return _refuse(inputs, impact, "MISSING_SUITE_ID", f"uncollectable={len(uncollectable)}:{uncollectable[0]}")
    if impact.status is ImpactStatus.REFUSED:
        return _conservative(inputs, impact, "UNKNOWN_PATH", "impact REFUSED", universe)
    if inputs.critical_omissions > 0:
        return _conservative(
            inputs, impact, "UNRESOLVED_NODE", f"critical_omissions={inputs.critical_omissions}", universe
        )
    for node in impact.affected:
        if node.kind == "CONSERVATIVE_UNKNOWN":
            return _conservative(inputs, impact, "UNKNOWN_PATH", node.node_id, universe)
        if node.node_id in inputs.modules_with_unknown_deps:
            return _conservative(inputs, impact, "UNRESOLVED_NODE", node.node_id, universe)
    selected: dict[str, str] = {n: "R2-MANDATORY" for n in inputs.mandatory_node_ids}
    reasons: list[SelectionReason] = []
    affected_suites = {a.node_id[len("suite:") :]: a for a in impact.affected if a.node_id.startswith("suite:")}
    touches_self = any(a.node_id.startswith(_SELF_SURFACE_PREFIXES) for a in impact.affected)
    for node_id in sorted(universe):
        module = _node_module(node_id)
        if module in affected_suites:
            witness = " > ".join(affected_suites[module].witness_path)
            reasons.append(SelectionReason(node_id=node_id, rule_id="R1-SURFACE", witness=witness))
            if node_id not in selected:
                selected[node_id] = "R2-POP-A" if node_id in inputs.pop_a_node_ids else "R1-SURFACE"
        elif touches_self and node_id in inputs.evol_manifest_node_ids and node_id not in selected:
            selected[node_id] = "R3-EVOL-VL3"
    if not selected:
        return _refuse(inputs, impact, "EMPTY_CATALOG", "empty selection forbidden")
    return _decision(SelectionStatus.SELECTED, inputs, impact, selected=selected, reasons=tuple(reasons))


def select_tests_for_impact(
    impact: ImpactAssessmentResult,
    *,
    catalog_node_ids: frozenset[str],
    mandatory_node_ids: frozenset[str],
    **extras: object,
) -> SelectionDecision:
    """Backwards-compatible entry; extras map onto ``SelectionInputs`` fields."""
    return select_tests(
        SelectionInputs(
            impact=impact,
            catalog_node_ids=catalog_node_ids,
            mandatory_node_ids=mandatory_node_ids,
            **extras,  # type: ignore[arg-type]
        )
    )
