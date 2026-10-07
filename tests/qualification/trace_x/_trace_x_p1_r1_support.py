# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P1-R1 strict durable child lineage admission SSOT (qualification-local)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from tests.qualification.trace_x._trace_x_p0_support import repo_root

TRACE_X_P1_R1_START_HEAD: Final[str] = "f31a326bb96b936e07d149c2ba8c75351d9117c4"
TRACE_X_P1_AUDITED_TRANSPORT_HEAD: Final[str] = (
    "097b8236817456377848885a324afa1044101009"
)

MANDATORY_FRZ_R1_IDS: Final[tuple[str, ...]] = ("FRZ-TRC-02",)

P1_BLOCKER_RESOLVED_ID: Final[str] = "P1-BLK-DEGRADED-LINEAGE-01"
P1_BLOCKER_RESOLUTION: Final[str] = "RESOLVED PENDING INDEPENDENT AUDIT"

ARCHITECTURE_DECISION_SUMMARY: Final[str] = (
    "child execution requires successful canonical durable "
    "ExecutionLineage parent→child admission"
)

STRICT_ADMISSION_EVIDENCE_TESTS: Final[tuple[str, ...]] = (
    "test_child_admission_unavailable_blocks_delegate_and_marks_degraded",
    "test_mark_degraded_unavailable_still_blocks_delegate",
    "test_sibling_after_failed_child_admission_may_execute_when_durable",
    "test_failed_child_admission_reconstruction_honest",
    "test_child_budget_released_after_failed_lineage_admission",
    "test_parent_identity_and_authority_restored_after_failed_admission",
    "test_conflicting_parent_still_blocks_child_delegate",
    "test_child_lineage_hook_auto_attached",
    "test_child_lineage_hook_auto_attached",
)


class R1GateResult(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class R1ReadinessStatus(StrEnum):
    READY_FOR_AUDIT = "READY FOR AUDIT"
    BLOCKED = "BLOCKED"


class FrzTrcR1Disposition(StrEnum):
    READY_FOR_INDEPENDENT_CLOSURE_REVIEW = "READY FOR INDEPENDENT CLOSURE REVIEW"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class R1EnterpriseAuditRow:
    area: str
    result: R1GateResult


FRZ_TRC_02_DISPOSITION: Final[FrzTrcR1Disposition] = (
    FrzTrcR1Disposition.READY_FOR_INDEPENDENT_CLOSURE_REVIEW
)

R1_READINESS: Final[R1ReadinessStatus] = R1ReadinessStatus.READY_FOR_AUDIT
P1_POST_R1_READINESS: Final[R1ReadinessStatus] = R1ReadinessStatus.READY_FOR_AUDIT

EXECUTED_CHILD_WITHOUT_DURABLE_PARENT_EDGE: Final[bool] = False

ENTERPRISE_AUDIT_MATRIX_R1: Final[tuple[R1EnterpriseAuditRow, ...]] = (
    R1EnterpriseAuditRow("durable child admission prerequisite", R1GateResult.PASS),
    R1EnterpriseAuditRow("failed admission blocks delegate", R1GateResult.PASS),
    R1EnterpriseAuditRow("degradation does not authorize execution", R1GateResult.PASS),
    R1EnterpriseAuditRow("mark_degraded failure blocks execution", R1GateResult.PASS),
    R1EnterpriseAuditRow("structural conflicts block execution", R1GateResult.PASS),
    R1EnterpriseAuditRow("successful admission permits child", R1GateResult.PASS),
    R1EnterpriseAuditRow("admission before delegate", R1GateResult.PASS),
    R1EnterpriseAuditRow("obsolete non_durable branch removed", R1GateResult.PASS),
    R1EnterpriseAuditRow("exactly-one lineage owner", R1GateResult.PASS),
    R1EnterpriseAuditRow("no RuntimeEvent topology fallback", R1GateResult.PASS),
    R1EnterpriseAuditRow("no causal-evidence topology fallback", R1GateResult.PASS),
    R1EnterpriseAuditRow("budget cleanup", R1GateResult.PASS),
    R1EnterpriseAuditRow("identity cleanup", R1GateResult.PASS),
    R1EnterpriseAuditRow("authority cleanup", R1GateResult.PASS),
    R1EnterpriseAuditRow("degradation semantics retained", R1GateResult.PASS),
    R1EnterpriseAuditRow("reconstruction truthfulness", R1GateResult.PASS),
    R1EnterpriseAuditRow("contracts over implementations", R1GateResult.PASS),
    R1EnterpriseAuditRow("strong typing", R1GateResult.PASS),
    R1EnterpriseAuditRow("pluginability", R1GateResult.PASS),
    R1EnterpriseAuditRow("tenant isolation P1-R1", R1GateResult.PASS),
    R1EnterpriseAuditRow("FRZ-TRC-02 readiness", R1GateResult.PASS),
)

TENANT_ISOLATION_AUDIT_R1: Final[dict[str, str]] = {
    "tenant_scope_applicable": "YES",
    "canonical_tenant_identity": "tenant_id",
    "tenant_owner": "ExecutionLineageAttemptScope / active execution identity",
    "propagation_path": "parent active execution → child identity → lineage admission scope",
    "state_isolation": "unchanged",
    "provider_config_isolation": "unchanged / not R1 scope",
    "evidence_trace_isolation": "strict lineage write before child execution",
    "async_recovery_continuity": "degraded attempt remains explicit; resume policy unchanged",
    "cross_tenant_path": "existing negative evidence",
    "fail_closed": "child admission failure blocks delegate",
    "result": "PASS",
}

_PRODUCTION_ROOT: Final[Path] = repo_root() / "intergrax"


def production_text(rel: str) -> str:
    return (_PRODUCTION_ROOT.parent / rel).read_text(encoding="utf-8")


def assert_obsolete_non_durable_machinery_removed() -> None:
    for rel in (
        "intergrax/runtime/execution/lineage/admission.py",
        "intergrax/runtime/execution/lineage/active_lineage.py",
        "intergrax/runtime/execution/child.py",
    ):
        text = production_text(rel)
        assert "mark_execution_lineage_non_durable" not in text
        assert "non_durable_execution_ids" not in text
        assert "non-durable lineage parent" not in text


def assert_child_admission_re_raises_after_degradation() -> None:
    admission = production_text("intergrax/runtime/execution/lineage/admission.py")
    assert "mark_attempt_lineage_degraded()" in admission
    assert "mark_execution_lineage_non_durable" not in admission
    degraded_idx = admission.index("mark_attempt_lineage_degraded()")
    raise_idx = admission.index("raise", degraded_idx)
    assert raise_idx > degraded_idx
