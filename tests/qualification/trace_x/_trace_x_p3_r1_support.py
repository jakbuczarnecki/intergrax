# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1 governed boundary v2 qualification SSOT (qualification-local)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.trace_x._trace_x_p0_support import repo_root

TRACE_X_P3_R1_START_HEAD: Final[str] = "72c1aca53663fe002faf2bb9bc4d5a14d2be92b1"
TRACE_X_P3_R1_R1_START_HEAD: Final[str] = "7bc5a7ec0fbacaf52558fdaf53384c20fa359e1b"

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

ENTERPRISE_AUDIT_MATRIX_GATE_IDS: Final[dict[str, tuple[str, ...]]] = {
    "Execution sole identity authority": ("TXP3R1R1-Q03", "TXP3R1R1-Q04", "TXP3R1R1-Q05"),
    "tenant isolation": (
        "TXP3R1R1-Q06",
        "TXP3R1R1-Q07",
        "TXP3R1R1-Q08",
        "TXP3R1R1-Q09",
        "TXP3R1R1-Q16",
        "TXP3R1R1-Q17",
    ),
    "FRZ-TRC-04 readiness": ("TXP3R1R1-Q23", "TXP3R1R1-Q03", "TXP3R1R1-Q09"),
    "FRZ-TRC-06 readiness": ("TXP3R1R1-Q24", "TXP3R1R1-Q06", "TXP3R1R1-Q16"),
}

TENANT_ISOLATION_AUDIT_P3_R1_R1: Final[dict[str, str]] = {
    "tenant_scope_applicable": "YES",
    "canonical_tenant_identity": "ActiveExecutionGovernanceIdentity.tenant_id",
    "tenant_owner": "existing Governance/runtime execution context",
    "propagation_path": (
        "active governance tenant → host validation → GovernedExecutionResultV2 "
        "→ ExecutionBoundaryEventV2 → ProofReceiptV2"
    ),
    "cross_tenant_path": "tenant A context + tenant B request → fail before provider",
    "fail_closed_behavior": "missing/blank/mismatched tenant → no provider effect",
    "adversarial_evidence": "TXP3R1R1-Q07..Q17; host test_p3_r1_r1_governed_identity_tenant_gate",
    "result": "PASS",
}
