# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from testing_support.change_safety.contracts.enums import SelectionStatus


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
