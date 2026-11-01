# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.contracts.impact import AffectedNode, ImpactAssessmentResult
from testing_support.change_safety.selection.selector import SelectionInputs


class SelectionFaultClass(StrEnum):
    UNKNOWN_PATH = "UNKNOWN_PATH"
    UNRESOLVED_NODE = "UNRESOLVED_NODE"
    DIGEST_MISMATCH = "DIGEST_MISMATCH"
    STALE_SNAPSHOT = "STALE_SNAPSHOT"
    ANALYZER_EXCEPTION = "ANALYZER_EXCEPTION"
    EMPTY_CATALOG = "EMPTY_CATALOG"
    MISSING_SUITE_ID = "MISSING_SUITE_ID"
    MISSING_METADATA = "MISSING_METADATA"


@dataclass(frozen=True)
class OracleExpectation:
    fault: SelectionFaultClass
    expected_status: SelectionStatus


FAULT_INJECTION_MATRIX: tuple[OracleExpectation, ...] = (
    OracleExpectation(SelectionFaultClass.UNKNOWN_PATH, SelectionStatus.CONSERVATIVE_FULL),
    OracleExpectation(SelectionFaultClass.UNRESOLVED_NODE, SelectionStatus.CONSERVATIVE_FULL),
    OracleExpectation(SelectionFaultClass.DIGEST_MISMATCH, SelectionStatus.REFUSED_INTEGRITY),
    OracleExpectation(SelectionFaultClass.STALE_SNAPSHOT, SelectionStatus.REFUSED_INTEGRITY),
    OracleExpectation(SelectionFaultClass.ANALYZER_EXCEPTION, SelectionStatus.CONSERVATIVE_FULL),
    OracleExpectation(SelectionFaultClass.EMPTY_CATALOG, SelectionStatus.REFUSED_INTEGRITY),
    OracleExpectation(SelectionFaultClass.MISSING_SUITE_ID, SelectionStatus.REFUSED_INTEGRITY),
    OracleExpectation(SelectionFaultClass.MISSING_METADATA, SelectionStatus.REFUSED_INTEGRITY),
)


def expected_status_for_fault(fault: SelectionFaultClass) -> SelectionStatus:
    for row in FAULT_INJECTION_MATRIX:
        if row.fault is fault:
            return row.expected_status
    raise KeyError(fault)


def _raising_analyzer() -> ImpactAssessmentResult:
    raise RuntimeError("injected analyzer fault")


def inject_selection_fault(fault: SelectionFaultClass, base: SelectionInputs) -> SelectionInputs:
    """Constructed fault oracle (S6/M-04): each class maps to exactly one fail-closed status."""
    assert base.impact is not None
    if fault is SelectionFaultClass.UNKNOWN_PATH:
        unknown = AffectedNode(
            node_id="unknown:path:injected/unknown.bin",
            kind="CONSERVATIVE_UNKNOWN",
            witness_path=("injected/unknown.bin",),
            rule_id="R-UNKNOWN-PATH",
        )
        return replace(base, impact=base.impact.model_copy(update={"affected": (*base.impact.affected, unknown)}))
    if fault is SelectionFaultClass.UNRESOLVED_NODE:
        return replace(base, critical_omissions=1)
    if fault is SelectionFaultClass.DIGEST_MISMATCH:
        return replace(base, expected_graph_digest="0" * 64)
    if fault is SelectionFaultClass.STALE_SNAPSHOT:
        return replace(base, expected_head_sha="f" * 40)
    if fault is SelectionFaultClass.ANALYZER_EXCEPTION:
        return replace(base, analyzer=_raising_analyzer)
    if fault is SelectionFaultClass.EMPTY_CATALOG:
        return replace(base, catalog_node_ids=frozenset())
    if fault is SelectionFaultClass.MISSING_SUITE_ID:
        return replace(base, required_suite_ids=frozenset({"tests/__injected_missing_suite__.py"}))
    if fault is SelectionFaultClass.MISSING_METADATA:
        return replace(base, impact=base.impact.model_copy(update={"ruleset_version": ""}))
    raise KeyError(fault)
