# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — shared ancestry anchors, fault matrix, invariant expectations."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

EE_B2_FINAL_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B2_FINAL_CHAOS_FAULT_MATRIX_CLOSURE.md"
)

REVALIDATION_COMMIT = "170460148744ac2324e67b7e992e71883e443561"
EE_FINAL_ARCH_COMMIT = "1c1005e2f66447e3f19f9aba8c0020b13c944b72"
NPSC5F_REQUALIFICATION_COMMIT = "cd0217ef0cbf2386f5f6134c30cfb80adf6ecddb"

ANCESTRY_ANCHORS: tuple[tuple[str, str], ...] = (
    ("INTEGRAx-CURRENT-HEAD-PLATFORM-REVALIDATION", REVALIDATION_COMMIT),
    ("EE-FINAL-ARCH", EE_FINAL_ARCH_COMMIT),
    ("NPSC-5F-R3-FINAL-REQUALIFICATION", NPSC5F_REQUALIFICATION_COMMIT),
)

REVALIDATION_AUDITED_HEAD = "edd44e2183c8ec78f6ca71697865d630ba221a9e"

EE_B2_BASE_TEST_MODULES: tuple[str, ...] = (
    "tests/unit/runtime/architecture/test_ee_b2_worker_fault_injection.py",
    "tests/unit/runtime/architecture/test_ee_b2_dependency_fault_injection.py",
    "tests/unit/runtime/architecture/test_ee_b2_capacity_saturation_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_cancellation_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_evidence_persistence_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_checkpoint_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_recovery_interruption.py",
    "tests/unit/runtime/architecture/test_ee_b2_child_partial_failure.py",
    "tests/unit/runtime/architecture/test_ee_b2_observability_export_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_compound_failure.py",
    "tests/unit/runtime/architecture/test_ee_b2_shutdown_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_chaos_persistence_wrapper_contract.py",
    "tests/unit/runtime/architecture/test_ee_b2_architecture_gate.py",
)

EE_B2_FINAL_TEST_MODULES: tuple[str, ...] = (
    "tests/unit/runtime/architecture/test_ee_b2_final_current_head_ancestry.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_fault_matrix.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_invariant_zero_counts.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_partial_recovery.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_compound_failures.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_resource_release.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_repeatability.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_architecture_gate.py",
)

FAULT_MATRIX_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "F-01",
        "ExecutionWorkPort",
        "concurrent_execution_work",
        "FAILED sibling isolation + no false success",
        "test_ee_b2_worker_fault_injection.py",
    ),
    (
        "F-02",
        "capacity admission",
        "LocalExecutionCapacityAdmission",
        "REJECT + permit release",
        "test_ee_b2_capacity_saturation_fault.py",
    ),
    (
        "F-03",
        "RuntimeEventPersistence append",
        "resolve_runtime_persistence_failure",
        "canonical persistence policy",
        "test_ee_b2_chaos_persistence_wrapper_contract.py",
    ),
    (
        "F-04",
        "mandatory evidence append",
        "RuntimeEventBus",
        "FAIL_CLOSED",
        "test_ee_b2_evidence_persistence_fault.py",
    ),
    (
        "F-05",
        "OTLP exporter",
        "observability export plugin",
        "degraded export; evidence persisted",
        "test_ee_b2_observability_export_fault.py",
    ),
    (
        "F-06",
        "dependency port",
        "ExecutionWorkPort",
        "typed ConnectionError; no authority mint",
        "test_ee_b2_dependency_fault_injection.py",
    ),
    (
        "F-07",
        "checkpoint store",
        "SQLiteTaskCheckpointStore",
        "StaleCheckpointWriteError",
        "test_ee_b2_checkpoint_fault.py",
    ),
    (
        "F-08",
        "retry policy",
        "NPSC-5E retry plane",
        "bounded attempts",
        "test_npsc5e_r1_final_retry_attempt_qualification.py",
    ),
    (
        "F-09",
        "recovery / stale CAS",
        "checkpoint store",
        "no blind resume",
        "test_ee_b2_recovery_interruption.py",
    ),
    (
        "F-10",
        "cancellation",
        "concurrent work + capacity",
        "cancel + release",
        "test_ee_b2_cancellation_fault.py",
    ),
    (
        "F-11",
        "fan-out slots",
        "resilient concurrent work",
        "partial failure; no sibling replay",
        "test_ee_b2_child_partial_failure.py",
    ),
    (
        "F-12",
        "compound faults",
        "primary failure authority",
        "worker/evidence primary",
        "test_ee_b2_compound_failure.py",
    ),
)

INVARIANT_EXPECTED_ZERO: dict[str, int] = {
    "false_success": 0,
    "duplicate_execution": 0,
    "capacity_leak": 0,
    "worker_leak": 0,
    "task_leak": 0,
    "hidden_retry": 0,
    "recovery_bypass": 0,
    "sealed_attempt_reopen": 0,
    "successful_sibling_replay": 0,
    "governance_bypass": 0,
    "authority_expansion": 0,
    "tenant_mutation": 0,
}
