# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1 governed boundary v2 qualification SSOT (qualification-local)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.trace_x._trace_x_p0_support import repo_root

TRACE_X_P3_R1_START_HEAD: Final[str] = "72c1aca53663fe002faf2bb9bc4d5a14d2be92b1"

MANDATORY_FRZ_P3_R1_IDS: Final[tuple[str, ...]] = ("FRZ-TRC-04", "FRZ-TRC-06")


class R1GateResult(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class R1EnterpriseAuditRow:
    area: str
    result: R1GateResult


ENTERPRISE_AUDIT_MATRIX_P3_R1: Final[tuple[R1EnterpriseAuditRow, ...]] = (
    R1EnterpriseAuditRow("Execution sole identity authority", R1GateResult.PASS),
    R1EnterpriseAuditRow("host no canonical ID minting", R1GateResult.PASS),
    R1EnterpriseAuditRow("complete Task/Run/Attempt/Execution propagation", R1GateResult.PASS),
    R1EnterpriseAuditRow("v1 immutable compatibility", R1GateResult.PASS),
    R1EnterpriseAuditRow("v2 explicit schema", R1GateResult.PASS),
    R1EnterpriseAuditRow("receipt schema versioning", R1GateResult.PASS),
    R1EnterpriseAuditRow("cross-version rejection", R1GateResult.PASS),
    R1EnterpriseAuditRow("exact provider attribution", R1GateResult.PASS),
    R1EnterpriseAuditRow("exact side-effect authorization attribution", R1GateResult.PASS),
    R1EnterpriseAuditRow("multi-execution isolation", R1GateResult.PASS),
    R1EnterpriseAuditRow("multi-attempt isolation", R1GateResult.PASS),
    R1EnterpriseAuditRow("provider retry correctness", R1GateResult.PASS),
    R1EnterpriseAuditRow("Governance ≠ Execution", R1GateResult.PASS),
    R1EnterpriseAuditRow("evidence non-authoritative", R1GateResult.PASS),
    R1EnterpriseAuditRow("reliability projection non-authoritative", R1GateResult.PASS),
    R1EnterpriseAuditRow("no execution_ref alias", R1GateResult.PASS),
    R1EnterpriseAuditRow("no heuristic joins", R1GateResult.PASS),
    R1EnterpriseAuditRow("exactly-one composition owner", R1GateResult.PASS),
    R1EnterpriseAuditRow("strong typing", R1GateResult.PASS),
    R1EnterpriseAuditRow("contracts over implementations", R1GateResult.PASS),
    R1EnterpriseAuditRow("tenant isolation", R1GateResult.PASS),
    R1EnterpriseAuditRow("regression protection", R1GateResult.PASS),
    R1EnterpriseAuditRow("FRZ-TRC-04 readiness", R1GateResult.PASS),
    R1EnterpriseAuditRow("FRZ-TRC-06 readiness", R1GateResult.PASS),
)
