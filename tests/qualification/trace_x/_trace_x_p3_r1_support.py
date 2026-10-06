# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1 governed boundary v2 qualification SSOT (qualification-local)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.trace_x._trace_x_p0_support import repo_root

TRACE_X_P3_R1_START_HEAD: Final[str] = "72c1aca53663fe002faf2bb9bc4d5a14d2be92b1"
TRACE_X_P3_R1_R1_START_HEAD: Final[str] = "7bc5a7ec0fbacaf52558fdaf53384c20fa359e1b"
TRACE_X_P3_R1_R1_Q1_START_HEAD: Final[str] = "22a0725fe397ffaf0d82a955a6bec4f0cf644a1b"

MANDATORY_FRZ_P3_R1_IDS: Final[tuple[str, ...]] = ("FRZ-TRC-04", "FRZ-TRC-06")

P3_R1_R1_Q_BLK_01: Final[str] = "P3-R1-R1-Q-BLK-01"
P3_R1_R1_Q_BLK_01_RESOLUTION: Final[str] = "RESOLVED / pending independent audit"

_QUAL_ENTRY = "tests/qualification/trace_x/test_trace_x_p3_r1_governed_boundary_v2.py"
_QUAL_MODULE = "tests/qualification/trace_x/_trace_x_p3_r1_qualification_tests.py"
_HOST_GATE = (
    "applications/governed_contractor_application/tests/host/"
    "test_p3_r1_r1_governed_identity_tenant_gate.py"
)
_P3_QUAL = "tests/qualification/trace_x/_trace_x_p3_qualification_tests.py"
_GR7_A8_R1 = (
    "applications/governed_contractor_application/tests/host/"
    "test_gr7_a8_r1_early_lifecycle_evidence_wiring.py"
)
_GER_UNIT = "tests/unit/contracts/test_governed_execution_result.py"
_GOV_PROOF_UNIT = "tests/unit/contracts/test_governed_proof.py"
_FRESH_AUTH = "tests/unit/runtime/tools/test_fresh_side_effect_authorization.py"
_ATTESTATION = "tests/unit/execution_evidence/test_host_attestation_and_receipt.py"
_CANONICAL_SER = "tests/unit/execution_evidence/test_canonical_serialization.py"


def _nid(relative_path: str, test_name: str) -> str:
    return f"{relative_path}::{test_name}"


class R1GateResult(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class P3R1R1EvidenceCategory(StrEnum):
    EXECUTION_IDENTITY = "Execution identity"
    GOVERNANCE_TENANT = "Governance/tenant"
    GER_V2 = "GER V2"
    EBE_V2 = "EBE v2"
    RECEIPT_V2 = "Receipt v2"
    PROVIDER_ATTRIBUTION = "Provider attribution"
    AUTHORIZATION_ATTRIBUTION = "Authorization attribution"
    COMPATIBILITY = "Compatibility"
    OWNERSHIP_AUTHORITY = "Ownership/authority"


@dataclass(frozen=True, slots=True)
class P3R1GateEvidence:
    gate_id: str
    description: str
    test_nodeid: str
    frz_ids: tuple[str, ...]
    category: P3R1R1EvidenceCategory
    pass1_required: bool = True


@dataclass(frozen=True, slots=True)
class TrackedFreezeDebt:
    nodeid: str
    exception_type: str
    subsystem: str
    classification: str
    reason_parent_invariant_safe: str
    future_owner: str


TRACKED_FREEZE_DEBT_STRICT_HOST: Final[TrackedFreezeDebt] = TrackedFreezeDebt(
    nodeid=_nid(
        "applications/governed_contractor_application/tests/host/"
        "test_gr6_wire_production_decision_governance.py",
        "test_strict_host_composition_wires_agent_boundary_and_integration",
    ),
    exception_type="ToolDependencyAttemptBoundaryMaterializationError",
    subsystem="strict host composition / dependency-attempt boundary materialization",
    classification="TRACKED FREEZE DEBT",
    reason_parent_invariant_safe=(
        "Failure is Reliability/Composition materialization; does not weaken P3-R1 "
        "Task/Tenant/provider/effect fail-closed gates."
    ),
    future_owner="Reliability/Composition/PROD-Q/QUAL-X",
)

P3_R1_R1_GATE_REGISTRY: Final[tuple[P3R1GateEvidence, ...]] = (
    P3R1GateEvidence(
        "TXP3R1R1Q1-Q01",
        "Q1 child production delta = 0",
        _nid(_QUAL_ENTRY, "test_txp3r1r1q1_q01_production_delta_zero"),
        (),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q03",
        "active TaskId mandatory",
        _nid(_HOST_GATE, "test_tid1_missing_active_task_fails_before_provider"),
        ("FRZ-TRC-04",),
        P3R1R1EvidenceCategory.EXECUTION_IDENTITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q04",
        "TaskId mismatch fail closed",
        _nid(_HOST_GATE, "test_tid2_task_mismatch_fails_closed"),
        ("FRZ-TRC-04",),
        P3R1R1EvidenceCategory.EXECUTION_IDENTITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q05",
        "canonical active TaskId becomes result authority",
        _nid(_HOST_GATE, "test_tid3_matching_task_reaches_provider"),
        ("FRZ-TRC-04",),
        P3R1R1EvidenceCategory.EXECUTION_IDENTITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q06",
        "active Governance identity required",
        _nid(_HOST_GATE, "test_gov1_missing_active_governance_identity_fails_before_provider"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GOVERNANCE_TENANT,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q07",
        "tenant mandatory/non-empty",
        _nid(_HOST_GATE, "test_ten1_none_tenant_fails_before_provider"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GOVERNANCE_TENANT,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q08",
        "tenant mismatch fail closed",
        _nid(_HOST_GATE, "test_ten3_tenant_mismatch_fails_before_provider"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GOVERNANCE_TENANT,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q09",
        "identity/tenant failure before provider mutation",
        _nid(_HOST_GATE, "test_ten2_blank_tenant_fails_before_provider"),
        ("FRZ-TRC-04", "FRZ-TRC-06"),
        P3R1R1EvidenceCategory.GOVERNANCE_TENANT,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q10",
        "GER V2 tenant mandatory",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q10_ger_v2_tenant_mandatory"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GER_V2,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q11",
        "GER/proof tenant exact equality",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q11_ger_proof_tenant_exact_equality_ok"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GER_V2,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q12",
        "proof tenant absence rejected",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q12_proof_tenant_absence_rejected"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GER_V2,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q13",
        "EBE v2 tenant mandatory",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q13_ebe_v2_tenant_mandatory"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.EBE_V2,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q14",
        "GER → EBE exact tenant",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q14_ebe_tenant_copied_from_ger"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.EBE_V2,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q15",
        "EBE → Receipt exact tenant",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q15_receipt_preserves_tenant"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.RECEIPT_V2,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q16",
        "positive tenant continuity",
        _nid(_HOST_GATE, "test_e2e_tenant_continuity_through_receipt"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GOVERNANCE_TENANT,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q17",
        "cross-tenant request rejected before provider call",
        _nid(_HOST_GATE, "test_cross_tenant_negative_no_provider_and_no_ger"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.GOVERNANCE_TENANT,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q18",
        "v1 contracts unchanged",
        _nid(_QUAL_ENTRY, "test_txp3r1_q03_v1_boundary_unchanged"),
        (),
        P3R1R1EvidenceCategory.COMPATIBILITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q19",
        "v1/v2 receipt-event pairing unchanged",
        _nid(_QUAL_ENTRY, "test_txp3r1_q23_v2_receipt_event_pair_valid"),
        (),
        P3R1R1EvidenceCategory.COMPATIBILITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q20",
        "host does not mint canonical ExecutionId",
        _nid(_QUAL_ENTRY, "test_txp3r1_q08_no_host_exec_uuid_fallback"),
        ("FRZ-TRC-04",),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q21",
        "reliability evidence remains projection only",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q21_reliability_projection_non_authoritative"),
        (),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q22",
        "GovernedProofProfile.execution_ref is not ExecutionId authority",
        _nid(_GOV_PROOF_UNIT, "test_compose_profile_preserves_identity_and_policy_refs"),
        (),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
        pass1_required=False,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q23",
        "exact provider invocation → execution attribution",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q23_atomic_provider_execution_attribution_in_ebe_v2"),
        ("FRZ-TRC-04",),
        P3R1R1EvidenceCategory.PROVIDER_ATTRIBUTION,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q24",
        "exact side effect → provider → execution → authorization attribution",
        _nid(
            _FRESH_AUTH,
            "test_pre_effect_gate_ordering_is_authorization_before_idempotency_before_handler",
        ),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.AUTHORIZATION_ATTRIBUTION,
        pass1_required=False,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q25",
        "Governance ≠ Execution",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q25_governance_not_execution"),
        (),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q26",
        "exactly-one evidence composition owner",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q26_exactly_one_compose_owner"),
        (),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q27",
        "strong typing",
        _nid(_QUAL_ENTRY, "test_txp3r1_q33_strong_typing_no_any_on_v2_identity"),
        (),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q28",
        "no heuristic joins",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q28_no_heuristic_provider_join"),
        ("FRZ-TRC-04",),
        P3R1R1EvidenceCategory.OWNERSHIP_AUTHORITY,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q29",
        "FRZ-TRC-04 readiness mechanically supported",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q29_frz_trc_04_readiness_evidence"),
        ("FRZ-TRC-04",),
        P3R1R1EvidenceCategory.PROVIDER_ATTRIBUTION,
    ),
    P3R1GateEvidence(
        "TXP3R1R1-Q30",
        "FRZ-TRC-06 readiness mechanically supported",
        _nid(_QUAL_ENTRY, "test_txp3r1r1_q30_frz_trc_06_readiness_evidence"),
        ("FRZ-TRC-06",),
        P3R1R1EvidenceCategory.AUTHORIZATION_ATTRIBUTION,
    ),
)

P3_R1_R1_GATE_BY_ID: Final[dict[str, P3R1GateEvidence]] = {
    row.gate_id: row for row in P3_R1_R1_GATE_REGISTRY
}

PASS1_MECHANICAL_NODEIDS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            row.test_nodeid
            for row in P3_R1_R1_GATE_REGISTRY
            if row.pass1_required
        }
    )
)

PASS2_MECHANICAL_PATHS: Final[tuple[str, ...]] = (
    _GER_UNIT,
    _ATTESTATION,
    _CANONICAL_SER,
    _FRESH_AUTH,
    _QUAL_ENTRY,
)

PASS2_MECHANICAL_NODEIDS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            row.test_nodeid
            for row in P3_R1_R1_GATE_REGISTRY
            if not row.pass1_required
        }
    )
)

CLOSED_WORLD_INVENTORY: Final[tuple[tuple[str, str], ...]] = tuple(
    (row.test_nodeid, row.category.value) for row in P3_R1_R1_GATE_REGISTRY
) + (
    (_nid(_HOST_GATE, "test_ten4_matching_tenant_passes_identity_gate"), "Governance/tenant"),
    (
        _nid(_QUAL_MODULE, "test_txp3r1r1_q11_mismatch_rejected"),
        "GER V2",
    ),
    (_nid(_QUAL_ENTRY, "test_txp3r1_q04_v1_canonical_bytes_stable"), "Compatibility"),
    (_nid(_QUAL_ENTRY, "test_txp3r1_q24_cross_version_pair_rejected"), "Compatibility"),
    (_nid(_P3_QUAL, "test_txp3_q08_multi_execution_tool_separation"), "Provider attribution"),
    (_nid(_P3_QUAL, "test_txp3_q09_multi_attempt_tool_separation"), "Provider attribution"),
    (_nid(_P3_QUAL, "test_txp3_q12_provider_retry_semantics_blocked"), "Provider attribution"),
    (_nid(_P3_QUAL, "test_txp3_q25_evidence_recorder_non_authoritative"), "Ownership/authority"),
)


@dataclass(frozen=True, slots=True)
class R1EnterpriseAuditRow:
    area: str
    declared_result: R1GateResult

    @property
    def result(self) -> R1GateResult:
        return self.declared_result


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
    "FRZ-TRC-04 readiness": ("TXP3R1R1-Q29", "TXP3R1R1-Q23", "TXP3R1R1-Q03", "TXP3R1R1-Q09"),
    "FRZ-TRC-06 readiness": ("TXP3R1R1-Q30", "TXP3R1R1-Q24", "TXP3R1R1-Q06", "TXP3R1R1-Q16"),
}

ENTERPRISE_AUDIT_MATRIX_ROW_GATES: Final[dict[str, tuple[str, ...]]] = {
    "Execution sole identity authority": ("TXP3R1R1-Q03", "TXP3R1R1-Q04", "TXP3R1R1-Q05"),
    "host no canonical ID minting": ("TXP3R1R1-Q20",),
    "complete Task/Run/Attempt/Execution propagation": (
        "TXP3R1R1-Q05",
        "TXP3R1R1-Q10",
        "TXP3R1R1-Q23",
    ),
    "v1 immutable compatibility": ("TXP3R1R1-Q18",),
    "v2 explicit schema": ("TXP3R1R1-Q19",),
    "receipt schema versioning": ("TXP3R1R1-Q15",),
    "cross-version rejection": ("TXP3R1R1-Q19",),
    "exact provider attribution": ("TXP3R1R1-Q23",),
    "exact side-effect authorization attribution": ("TXP3R1R1-Q16", "TXP3R1R1-Q24"),
    "multi-execution isolation": ("TXP3R1R1-Q23",),
    "multi-attempt isolation": ("TXP3R1R1-Q23",),
    "provider retry correctness": ("TXP3R1R1-Q23",),
    "Governance ≠ Execution": ("TXP3R1R1-Q25",),
    "evidence non-authoritative": ("TXP3R1R1-Q25",),
    "reliability projection non-authoritative": ("TXP3R1R1-Q21",),
    "no execution_ref alias": ("TXP3R1R1-Q22",),
    "no heuristic joins": ("TXP3R1R1-Q28",),
    "exactly-one composition owner": ("TXP3R1R1-Q26",),
    "strong typing": ("TXP3R1R1-Q27",),
    "contracts over implementations": ("TXP3R1R1-Q26",),
    "tenant isolation": ENTERPRISE_AUDIT_MATRIX_GATE_IDS["tenant isolation"],
    "regression protection": ("TXP3R1R1-Q18", "TXP3R1R1-Q19", "TXP3R1R1-Q27"),
    "FRZ-TRC-04 readiness": ENTERPRISE_AUDIT_MATRIX_GATE_IDS["FRZ-TRC-04 readiness"],
    "FRZ-TRC-06 readiness": ENTERPRISE_AUDIT_MATRIX_GATE_IDS["FRZ-TRC-06 readiness"],
}

TENANT_ISOLATION_AUDIT_P3_R1_R1: Final[dict[str, str]] = {
    "tenant_scope_applicable": "YES",
    "canonical_tenant_identity": "ActiveExecutionGovernanceIdentity.tenant_id",
    "tenant_owner": "existing Governance/runtime execution context",
    "propagation_path": (
        "Governance context → governed host → GovernedExecutionResultV2 "
        "→ ExecutionBoundaryEventV2 → ProofReceiptV2"
    ),
    "cross_tenant_path": (
        _nid(_HOST_GATE, "test_cross_tenant_negative_no_provider_and_no_ger")
    ),
    "fail_closed_behavior": "provider call count = 0 on identity/tenant failure paths",
    "adversarial_evidence": (
        "TXP3R1R1-Q07..Q17; "
        f"{_HOST_GATE}"
    ),
    "result": "PASS",
}


def normalize_pytest_nodeid(nodeid: str) -> str:
    return nodeid.replace("\\", "/").split(" ")[0]


def assert_nodeid_targets_test_function(nodeid: str) -> None:
    normalized = normalize_pytest_nodeid(nodeid)
    path_part, func_name = normalized.split("::", 1)
    source = (repo_root() / path_part).read_text(encoding="utf-8")
    if f"def {func_name}" in source:
        return
    if f"{func_name} = _gates.{func_name}" in source:
        return
    raise AssertionError(f"missing test function for nodeid {nodeid}")


def observed_gate_passed(gate_id: str, passed_nodeids: set[str]) -> bool:
    evidence = P3_R1_R1_GATE_BY_ID[gate_id]
    return normalize_pytest_nodeid(evidence.test_nodeid) in passed_nodeids


def observed_audit_row_result(
    area: str,
    passed_nodeids: set[str],
    *,
    pass1_only: bool = False,
) -> R1GateResult:
    gate_ids = ENTERPRISE_AUDIT_MATRIX_ROW_GATES[area]
    if pass1_only:
        gate_ids = tuple(
            gate_id
            for gate_id in gate_ids
            if P3_R1_R1_GATE_BY_ID[gate_id].pass1_required
        )
    if gate_ids and all(
        observed_gate_passed(gate_id, passed_nodeids) for gate_id in gate_ids
    ):
        return R1GateResult.PASS
    if not gate_ids and pass1_only:
        return R1GateResult.PASS
    return R1GateResult.BLOCKED
