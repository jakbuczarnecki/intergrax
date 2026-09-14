# © Artur Czarnecki. All rights reserved.

"""Frozen mandatory pytest suite declarations (canonical catalog SSOT)."""

from __future__ import annotations

from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)

from testing_support.execution_qualification.embedded_harness_kexpr import (
    R2_H2_Q1_EMBEDDED_HARNESS_KEXPR,
)

NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/test_npsc5e_r2_h2_q1_frozen_regression_closure.py"
)

NPSC5E_R2_H2_Q1_EMBEDDED_PREDECESSOR_LABELS: tuple[str, ...] = (
    "R1 Final",
    "R2 Original",
    "R2-H1",
    "R2-H2",
    "P0A",
    "DG_001 lineage",
    "NPSC-5D Final",
    "HITL R3",
    "NPSC-5A",
    "NPSC-5B",
    "NPSC-5C",
    "Attempt lifecycle",
    "Child execution",
    "Terminal",
    "Cancellation",
    "Checkpoint store",
    "Long-running",
)

NPSC5E_R3_FINAL_MANDATORY: FrozenPytestSuiteSource = (
    (
        "R1 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py"
        ],
    ),
    (
        "R2 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py"
        ],
    ),
    (
        "R3 implementation gate",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py"
        ],
    ),
    (
        "P0A",
        [
            "tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py"
        ],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5A",
        ["tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py"],
    ),
    (
        "NPSC-5B Final",
        [
            "tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"
        ],
    ),
    (
        "NPSC-5C",
        ["tests/unit/runtime/architecture/test_npsc5c_decision_execution_e2e.py"],
    ),
    (
        "NPSC-5D Final",
        [
            "tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"
        ],
    ),
    (
        "HITL R3",
        ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"],
    ),
    (
        "Attempt lifecycle",
        [
            "tests/unit/runtime/execution/test_attempt_lifecycle.py",
            "tests/unit/runtime/execution/test_attempt_lifecycle_durability_gate.py",
            "tests/conformance/runtime/durability/test_attempt_lifecycle.py",
        ],
    ),
    (
        "Child execution",
        [
            "tests/unit/runtime/execution/test_child_execution.py",
            "tests/unit/runtime/execution/authority/test_child_execution_authority_policy.py",
        ],
    ),
    (
        "Terminal",
        ["tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py"],
    ),
    (
        "Cancellation",
        [
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
            "-k",
            "not survives_process_restart",
            "tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py",
            "tests/unit/applications/test_task_control_governed_resume.py",
        ],
    ),
    ("Checkpoint store", ["tests/unit/runtime/long_running/test_checkpoint_store.py"]),
    (
        "Long-running",
        [
            "tests/unit/runtime/long_running/test_pcm_scheduler_integrity.py",
            "tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py",
            "tests/unit/runtime/long_running/test_runtime_checkpoint.py",
            "tests/unit/runtime/long_running/test_resume_planner.py",
            "tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py",
            "tests/unit/runtime/long_running/test_p0c3_recovery_state_authority.py",
        ],
    ),
    ("Fan-out", ["tests/unit/agent_distribution/test_bounded_multi_agent_fanout.py"]),
)

NPSC5E_R2_FINAL_MANDATORY: FrozenPytestSuiteSource = (
    (
        "R1 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py"
        ],
    ),
    (
        "R2 Original",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r2_checkpoint_durable_resume_hardening.py"
        ],
    ),
    (
        "R2-H1",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r2_h1_authority_stale_checkpoint_closure.py"
        ],
    ),
    (
        "R2-H2",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r2_h2_checkpoint_revision_stale_writer_protection.py"
        ],
    ),
    (
        "R2-H2-Q1",
        [
            NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH,
            "-k",
            R2_H2_Q1_EMBEDDED_HARNESS_KEXPR,
        ],
    ),
    (
        "P0A",
        [
            "tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py"
        ],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        [
            "tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"
        ],
    ),
    (
        "HITL R3",
        ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"],
    ),
    (
        "NPSC-5A",
        ["tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py"],
    ),
    (
        "NPSC-5B",
        [
            "tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"
        ],
    ),
    (
        "NPSC-5C",
        ["tests/unit/runtime/architecture/test_npsc5c_decision_execution_e2e.py"],
    ),
    (
        "Attempt lifecycle",
        [
            "tests/unit/runtime/execution/test_attempt_lifecycle.py",
            "tests/unit/runtime/execution/test_attempt_lifecycle_durability_gate.py",
            "tests/conformance/runtime/durability/test_attempt_lifecycle.py",
        ],
    ),
    (
        "Child execution",
        [
            "tests/unit/runtime/execution/test_child_execution.py",
            "tests/unit/runtime/execution/authority/test_child_execution_authority_policy.py",
        ],
    ),
    (
        "Terminal",
        ["tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py"],
    ),
    (
        "Cancellation",
        [
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
            "-k",
            "not survives_process_restart",
            "tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py",
            "tests/unit/applications/test_task_control_governed_resume.py",
        ],
    ),
    ("Checkpoint store", ["tests/unit/runtime/long_running/test_checkpoint_store.py"]),
    (
        "Long-running",
        [
            "tests/unit/runtime/long_running/test_pcm_scheduler_integrity.py",
            "tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py",
            "tests/unit/runtime/long_running/test_runtime_checkpoint.py",
            "tests/unit/runtime/long_running/test_resume_planner.py",
            "tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py",
            "tests/unit/runtime/long_running/test_p0c3_recovery_state_authority.py",
        ],
    ),
)

NPSC5E_FINAL_MANDATORY: FrozenPytestSuiteSource = (
    (
        "NPSC-5E recovery plane (R1+R2+R3 finals + section 94)",
        [
            "tests/unit/runtime/architecture/"
            "test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py",
        ],
    ),
)

NPSC5F_R1_FINAL_MANDATORY: FrozenPytestSuiteSource = (
    (
        "R1 implementation gate",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_commit_tenant_integrity.py"
        ],
    ),
    (
        "NPSC-5F P0 gate",
        [
            "tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py"
        ],
    ),
    ("Runtime events suites", ["tests/unit/runtime/events/"]),
    (
        "Runtime observability suites",
        ["tests/unit/runtime/observability/"],
    ),
    (
        "NPSC-5E Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
        ],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        [
            "tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"
        ],
    ),
)

NPSC5F_R2_FINAL_MANDATORY: FrozenPytestSuiteSource = (
    (
        "R2 implementation gate",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r2_journal_completeness_ordering.py"
        ],
    ),
    (
        "R1 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py"
        ],
    ),
    (
        "NPSC-5F P0 gate",
        [
            "tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py"
        ],
    ),
    ("Runtime events suites", ["tests/unit/runtime/events/"]),
    (
        "Runtime observability suites",
        ["tests/unit/runtime/observability/"],
    ),
    (
        "TRACE-ASOF",
        [
            "tests/unit/runtime/events/test_execution_position_asof.py",
            "tests/unit/runtime/events/test_asof_projection.py",
        ],
    ),
    (
        "TRACE-BITEMP",
        [
            "tests/unit/contracts/test_bitemporal_revision_ordering.py",
            "tests/unit/contracts/test_bitemporal_knowledge.py",
        ],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
    (
        "NPSC-5E Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
        ],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        [
            "tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"
        ],
    ),
    (
        "NPSC-5B Final",
        [
            "tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"
        ],
    ),
    (
        "R2 drift classifier",
        ["tests/unit/testing_support/test_npsc5f_r2_protected_drift.py"],
    ),
)

NPSC5F_R3_FINAL_MANDATORY: FrozenPytestSuiteSource = (
    (
        "R3 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r3_governed_evidence_export.py"],
    ),
    (
        "R2 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r2_final_journal_completeness_ordering.py"
        ],
    ),
    (
        "R1 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py"
        ],
    ),
    (
        "NPSC-5F P0 gate",
        [
            "tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py"
        ],
    ),
    ("Runtime events suites", ["tests/unit/runtime/events/"]),
    (
        "Runtime observability suites",
        ["tests/unit/runtime/observability/"],
    ),
    (
        "Journal export suites",
        ["tests/unit/runtime/observability/test_journal_export.py"],
    ),
    (
        "Export boundary suites",
        [
            "tests/unit/runtime/observability/test_export_boundary.py",
            "tests/unit/runtime/observability/test_export_boundary_contracts.py",
        ],
    ),
    (
        "TRACE-ASOF",
        [
            "tests/unit/runtime/events/test_execution_position_asof.py",
            "tests/unit/runtime/events/test_asof_projection.py",
        ],
    ),
    (
        "TRACE-BITEMP",
        [
            "tests/unit/contracts/test_bitemporal_revision_ordering.py",
            "tests/unit/contracts/test_bitemporal_knowledge.py",
        ],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
    (
        "NPSC-5E Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
        ],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        [
            "tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"
        ],
    ),
    (
        "R3 drift classifier",
        ["tests/unit/testing_support/test_npsc5f_r3_protected_drift.py"],
    ),
)

NPSC5E_R1_FINAL_MANDATORY: FrozenPytestSuiteSource = (
    (
        "NPSC-5E R1 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py"
        ],
    ),
)

NPSC5F_R4_MANDATORY_REGRESSION_SUITES: FrozenPytestSuiteSource = (
    (
        "R4 implementation gate",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_asof_bitemporal.py",
        ],
    ),
    (
        "R3 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r3_final_governed_evidence_export.py",
        ],
    ),
    (
        "R2 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r2_final_journal_completeness_ordering.py",
        ],
    ),
    (
        "R1 Final",
        [
            "tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py",
        ],
    ),
    (
        "NPSC-5F P0",
        [
            "tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py",
        ],
    ),
    (
        "TRACE-ASOF",
        [
            "tests/unit/runtime/events/test_execution_position_asof.py",
            "tests/unit/runtime/events/test_asof_projection.py",
        ],
    ),
    (
        "TRACE-BITEMP",
        [
            "tests/unit/contracts/test_bitemporal_revision_ordering.py",
            "tests/unit/contracts/test_bitemporal_knowledge.py",
            "tests/unit/runtime/observability/test_knowledge_reconstruction.py",
        ],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5E Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py",
        ],
    ),
    (
        "NPSC-5D Final",
        [
            "tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py",
        ],
    ),
    (
        "NPSC-5C",
        [
            "tests/unit/runtime/architecture/test_npsc5c_coordination_intent_gate.py",
            "tests/unit/runtime/architecture/test_npsc5c_decision_projection_gate.py",
        ],
    ),
    (
        "NPSC-5B Final",
        [
            "tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py",
        ],
    ),
    (
        "NPSC-5A",
        [
            "tests/unit/runtime/architecture/test_npsc5a_multi_agent_coordination_gate.py"
        ],
    ),
    (
        "R4 Final drift sentinel",
        ["tests/unit/testing_support/test_npsc5f_r4_final_protected_drift.py"],
    ),
)


def _npsc5f_final_recovery_pytest_targets() -> list[str]:
    from testing_support.execution_qualification.final_semantic_pytest import (
        npsc5f_final_recovery_pytest_arguments,
    )

    paths = (
        "tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py",
        "tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py",
        "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py",
    )
    return list(npsc5f_final_recovery_pytest_arguments(paths))


_NPSC5F_FINAL_EXTRA_SUITES: FrozenPytestSuiteSource = (
    (
        "Recovery",
        _npsc5f_final_recovery_pytest_targets(),
    ),
    (
        "NPSC-5E Final",
        [
            "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py",
        ],
    ),
    (
        "HITL R3",
        ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"],
    ),
    (
        "Child execution",
        [
            "tests/unit/runtime/execution/test_child_execution.py",
            "tests/unit/runtime/execution/authority/test_child_execution_authority_policy.py",
        ],
    ),
    (
        "Checkpoint",
        [
            "tests/unit/runtime/long_running/test_checkpoint_store.py",
            "tests/unit/runtime/long_running/test_runtime_checkpoint.py",
        ],
    ),
    (
        "Retry",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py",
        ],
    ),
    (
        "Cancellation",
        [
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
            "tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py",
        ],
    ),
    (
        "Evidence",
        [
            "tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py",
            "tests/unit/runtime/architecture/test_npsc5f_enterprise_evidence_certification.py",
            "tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_persistence_boundary_resignoff.py",
            "tests/unit/runtime/events/test_evidence_persistence_boundary.py",
        ],
    ),
    (
        "NPSC-5D Final",
        [
            "tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py",
        ],
    ),
    (
        "NPSC-5F Final drift sentinel",
        ["tests/unit/testing_support/test_npsc5f_final_protected_drift.py"],
    ),
)

NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES: FrozenPytestSuiteSource = (
    *NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
    *_NPSC5F_FINAL_EXTRA_SUITES,
)

__all__ = [
    "NPSC5E_FINAL_MANDATORY",
    "NPSC5E_R1_FINAL_MANDATORY",
    "NPSC5E_R2_FINAL_MANDATORY",
    "NPSC5E_R2_H2_Q1_EMBEDDED_PREDECESSOR_LABELS",
    "NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH",
    "NPSC5E_R3_FINAL_MANDATORY",
    "NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES",
    "NPSC5F_R1_FINAL_MANDATORY",
    "NPSC5F_R2_FINAL_MANDATORY",
    "NPSC5F_R3_FINAL_MANDATORY",
    "NPSC5F_R4_MANDATORY_REGRESSION_SUITES",
]
