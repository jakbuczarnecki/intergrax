# © Artur Czarnecki. All rights reserved.

"""SESSION-01 Q1..Q20 reproducible evidence catalog (import/map only — no duplicate proofs)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Session01QEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]
    architecture_reason_na: str | None = None


def _nid(relative_path: str, test_name: str) -> str:
    return f"tests/{relative_path}::{test_name}"


SESSION_01_Q_CATALOG: tuple[Session01QEvidence, ...] = (
    Session01QEvidence(
        "Q1",
        "Canonical pause through ExecutionContinuationPort",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r2_canonical_pause_resume.py",
                "test_request_pause_creates_pause_requested_revision_one",
            ),
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r2_canonical_pause_resume.py",
                "test_approve_and_resume_spine",
            ),
        ),
    ),
    Session01QEvidence(
        "Q2",
        "Task/Human state projection only",
        (
            _nid(
                "unit/runtime/human/test_gr5_r3_continuation_projection.py",
                "test_task_only_human_response_does_not_resume_canonical",
            ),
            _nid(
                "unit/runtime/human/test_gr5_r3_r1_canonical_first_atomic_projection.py",
                "test_canonical_success_call_order",
            ),
        ),
    ),
    Session01QEvidence(
        "Q3",
        "Durable continuation survives full recomposition",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r5_restart_exact_identity.py",
                "test_approve_after_restart_same_ids",
            ),
            _nid(
                "unit/mp4r7/test_enterprise_integration_qualification.py",
                "test_mp4r7_process_restart_resume",
            ),
        ),
    ),
    Session01QEvidence(
        "Q4",
        "Identity preserved across recomposition",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r5_restart_exact_identity.py",
                "test_no_new_attempt_or_execution_id_on_qualify",
            ),
            _nid(
                "unit/mp4r7/test_enterprise_integration_qualification.py",
                "test_mp4r7_same_human_review_authorizes_continuation",
            ),
        ),
    ),
    Session01QEvidence(
        "Q5",
        "Duplicate resume safe",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r2_canonical_pause_resume.py",
                "test_double_resolution_and_double_resume_blocked",
            ),
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r5_restart_exact_identity.py",
                "test_double_resume_after_restart_one_wins",
            ),
        ),
    ),
    Session01QEvidence(
        "Q6",
        "Stale revision fails",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r2_canonical_pause_resume.py",
                "test_stale_resolution_and_resume",
            ),
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r5_restart_exact_identity.py",
                "test_stale_cas_after_restart",
            ),
        ),
    ),
    Session01QEvidence(
        "Q7",
        "Wrong ExecutionId fails",
        (
            _nid(
                "unit/runtime/architecture/test_mp4r3_execution_continuation_integration_gates.py",
                "test_mp4r3_identity_mismatch_qualification",
            ),
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r5_restart_exact_identity.py",
                "test_pointer_identity_mismatch_fail_closed",
            ),
        ),
    ),
    Session01QEvidence(
        "Q8",
        "Wrong tenant fails",
        (
            _nid(
                "unit/runtime/architecture/test_ee_b3_a_cross_tenant_execution_gate.py",
                "test_ee_b3_a_checkpoint_resume_rejects_cross_tenant",
            ),
        ),
    ),
    Session01QEvidence(
        "Q9",
        "Missing continuation fails closed",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r5_restart_exact_identity.py",
                "test_recovery_handle_not_found_fail_closed",
            ),
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r4_r1_current_continuation_episode.py",
                "test_store_query_failure_fails_closed",
            ),
        ),
    ),
    Session01QEvidence(
        "Q10",
        "Checkpoint alone cannot resume execution",
        (
            _nid(
                "unit/runtime/human/test_gr5_r3_continuation_projection.py",
                "test_clear_pause_does_not_establish_canonical_resume",
            ),
            _nid(
                "unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py",
                "test_final_authority_missing_scenario",
            ),
        ),
    ),
    Session01QEvidence(
        "Q11",
        "Projection can be rebuilt",
        (
            _nid(
                "unit/runtime/human/test_gr5_r3_continuation_projection.py",
                "test_canonical_resumed_repairs_task_projection",
            ),
            _nid(
                "unit/runtime/human/test_gr5_r3_r1_canonical_first_atomic_projection.py",
                "test_reprojection_after_projection_failure",
            ),
        ),
    ),
    Session01QEvidence(
        "Q12",
        "Human approval durability/recovery",
        (
            _nid(
                "unit/runtime/human/test_g5b_hitl_resolution.py",
                "test_valid_active_pause_approve_persists_canonical_resolution",
            ),
            _nid(
                "unit/runtime/human/test_idt_fix_d_execution_identity_closure.py",
                "test_d1_distinct_identity_human_decision_persistence",
            ),
        ),
    ),
    Session01QEvidence(
        "Q13",
        "Governed grant scoped correctly after restart",
        (
            _nid(
                "unit/runtime/human/test_g5c2b1_governed_continuation_grant.py",
                "test_run_mismatch_fails_closed",
            ),
            _nid(
                "unit/runtime/human/test_g5b_hitl_resolution.py",
                "test_declarative_stale_response_cannot_create_grant_for_new_pending",
            ),
        ),
    ),
    Session01QEvidence(
        "Q14",
        "Q20 Nexus production-like continuation path",
        (
            _nid(
                "unit/runtime/governance/test_gr1_execution_identity_rebinding.py",
                "test_nexus_intake_governed_approval_without_nexus_ae_forwarding",
            ),
        ),
    ),
    Session01QEvidence(
        "Q15",
        "Agent checkpoint cannot mint Execution authority",
        (
            _nid(
                "unit/agents/persistence/test_pcm_compensation_coordination.py",
                "test_b1_no_unfenced_terminal_mutation_api",
            ),
            _nid(
                "unit/agents/persistence/test_checkpoint_wiring.py",
                "test_inject_acp_checkpoint_metadata_when_session_enabled",
            ),
        ),
    ),
    Session01QEvidence(
        "Q16",
        "Task checkpoint cannot override continuation lifecycle",
        (
            _nid(
                "unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py",
                "test_final_terminal_scenario",
            ),
        ),
    ),
    Session01QEvidence(
        "Q17",
        "Custom continuation provider",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r5_restart_exact_identity.py",
                "test_custom_durable_provider_qualification",
            ),
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r2_canonical_pause_resume.py",
                "test_replaceable_implementation_and_falsey_store_preserved",
            ),
        ),
    ),
    Session01QEvidence(
        "Q18",
        "Custom checkpoint provider where applicable",
        (
            _nid(
                "unit/applications/test_acp_checkpoint_host_wiring.py",
                "test_resolve_host_agent_checkpoint_store_uses_explicit_store",
            ),
        ),
    ),
    Session01QEvidence(
        "Q19",
        "Concurrent resume deterministic",
        (
            _nid(
                "unit/runtime/execution/continuation/test_gr5_r2_canonical_pause_resume.py",
                "test_concurrent_resume_max_one_success",
            ),
        ),
    ),
    Session01QEvidence(
        "Q20",
        "No global mutable session/checkpoint authority",
        (
            _nid(
                "unit/runtime/architecture/test_mp4r3_execution_continuation_integration_gates.py",
                "test_mp4r3_no_duplicate_continuation_lifecycle_authority",
            ),
            _nid(
                "unit/runtime/architecture/test_mp4r3_execution_continuation_integration_gates.py",
                "test_mp4r3_no_multiplayer_continuation_repository",
            ),
        ),
    ),
)

SESSION_01_MANDATORY_PYTEST_TARGETS: tuple[str, ...] = tuple(
    dict.fromkeys(
        node_id.rsplit("::", 1)[0]
        for entry in SESSION_01_Q_CATALOG
        for node_id in entry.pytest_node_ids
    )
)

__all__ = [
    "SESSION_01_MANDATORY_PYTEST_TARGETS",
    "SESSION_01_Q_CATALOG",
    "Session01QEvidence",
]
