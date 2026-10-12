# © Artur Czarnecki. All rights reserved.

"""TENANT-X-FINAL composed adversarial negatives — reuse R1–R4 SSOT only."""

from __future__ import annotations

from tests.qualification.tenant_x._r1_adversarial import (
    assert_functional_evidence_rejects_missing_tenant,
    assert_governance_tenant_matches_runtime_envelope,
    assert_root_admission_missing_tenant_fails_closed,
    assert_runtime_request_tenant_a_cannot_become_b,
    assert_tool_scope_tenant_follows_inspection_scope,
)
from tests.qualification.tenant_x._r2_adversarial import (
    assert_cross_tenant_credential_ref_rejects,
    assert_missing_tenant_on_credential_scope_rejects,
    assert_reference_executor_rejects_evidence_tenant_mismatch,
    assert_vector_scope_cross_tenant_rejects,
)
from tests.qualification.tenant_x._r3_adversarial import (
    assert_agent_checkpoint_cross_tenant_load_rejects,
    assert_checkpoint_recovery_cross_tenant_rejects,
    assert_missing_tenant_resume_rejects_no_global,
    assert_partial_recovery_tenant_mismatch_rejects,
    assert_retry_resume_preserves_tenant_a,
    assert_tenant_b_cannot_load_tenant_a_durable_checkpoint,
)
from tests.qualification.tenant_x._r4_adversarial import (
    assert_background_blank_transport_rejects,
    assert_background_transport_task_tenant_mismatch_rejects,
    assert_context_llm_cross_tenant_attribution_rejects,
    assert_context_view_request_authority_mismatch_rejects,
    assert_context_view_request_membership_mismatch_rejects,
    assert_functional_evidence_missing_tenant_rejects,
    assert_vendor_queue_request_tenant_mismatch_rejects,
    assert_vendor_worker_payload_job_tenant_mismatch_rejects,
)


def run_final_authority_isolation_negatives() -> None:
    assert_runtime_request_tenant_a_cannot_become_b()
    assert_root_admission_missing_tenant_fails_closed()
    assert_governance_tenant_matches_runtime_envelope()
    assert_context_view_request_authority_mismatch_rejects()


def run_final_state_isolation_negatives() -> None:
    assert_tenant_b_cannot_load_tenant_a_durable_checkpoint()
    assert_agent_checkpoint_cross_tenant_load_rejects()


def run_final_provider_config_isolation_negatives() -> None:
    assert_vector_scope_cross_tenant_rejects()


def run_final_evidence_isolation_negatives() -> None:
    assert_context_llm_cross_tenant_attribution_rejects()
    assert_reference_executor_rejects_evidence_tenant_mismatch()


def run_final_credential_isolation_negatives() -> None:
    assert_cross_tenant_credential_ref_rejects()


def run_final_recovery_isolation_negatives() -> None:
    assert_checkpoint_recovery_cross_tenant_rejects()
    assert_partial_recovery_tenant_mismatch_rejects()


def run_final_child_async_isolation_negatives() -> None:
    assert_background_transport_task_tenant_mismatch_rejects()
    assert_vendor_queue_request_tenant_mismatch_rejects()
    assert_vendor_worker_payload_job_tenant_mismatch_rejects()
    assert_context_view_request_membership_mismatch_rejects()


def run_final_missing_tenant_fail_closed_negatives() -> None:
    assert_missing_tenant_on_credential_scope_rejects()
    assert_missing_tenant_resume_rejects_no_global()
    assert_functional_evidence_rejects_missing_tenant()
    assert_functional_evidence_missing_tenant_rejects()
    assert_background_blank_transport_rejects()


def run_final_trace_preservation_negatives() -> None:
    assert_retry_resume_preserves_tenant_a()
    assert_tool_scope_tenant_follows_inspection_scope()


def run_final_composed_adversarial_negatives() -> None:
    run_final_authority_isolation_negatives()
    run_final_state_isolation_negatives()
    run_final_provider_config_isolation_negatives()
    run_final_evidence_isolation_negatives()
    run_final_credential_isolation_negatives()
    run_final_recovery_isolation_negatives()
    run_final_child_async_isolation_negatives()
    run_final_missing_tenant_fail_closed_negatives()
    run_final_trace_preservation_negatives()
