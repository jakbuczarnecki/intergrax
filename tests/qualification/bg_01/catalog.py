# © Artur Czarnecki. All rights reserved.

"""BG-01 Q1..Q15 evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Bg01QEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]


def _nid(test_file: str, test_name: str) -> str:
    return f"tests/qualification/bg_01/{test_file}::{test_name}"


BG_01_Q_CATALOG: tuple[Bg01QEvidence, ...] = (
    Bg01QEvidence(
        "BG-Q1",
        "All active production background task-execution consumers reach HostTaskExecutionPort",
        (
            _nid("test_bg_01_gates.py", "test_bg_q1_worker_invokes_host_task_execution_port"),
            _nid("test_bg_01_gates.py", "test_bg_q1_production_background_execution_surfaces_use_host_port"),
        ),
    ),
    Bg01QEvidence(
        "BG-Q2",
        "Canonical identity (tenant, task, run, attempt) preserved through worker execute_payload",
        (
            _nid("test_bg_01_gates.py", "test_bg_q2_identity_forwarded_to_host_execution"),
            _nid("test_bg_01_gates.py", "test_bg_q2_full_canonical_identity_on_task_and_host_kwargs"),
        ),
    ),
    Bg01QEvidence(
        "BG-Q3",
        "Tenant provenance fail-closed before execution",
        (_nid("test_bg_01_gates.py", "test_bg_q3_tenant_mismatch_blocks_execution"),),
    ),
    Bg01QEvidence(
        "BG-Q4",
        "Background host-task path uses root execution launcher governance",
        (_nid("test_bg_01_gates.py", "test_bg_q4_worker_host_stack_uses_root_launcher"),),
    ),
    Bg01QEvidence(
        "BG-Q5",
        "Duplicate delivery with idempotency key does not double-invoke handler",
        (_nid("test_bg_01_gates.py", "test_bg_q5_idempotent_logical_task_single_handler_invoke"),),
    ),
    Bg01QEvidence(
        "BG-Q6",
        "Transport retry policy distinct from execution attempt lifecycle owner",
        (
            _nid("test_bg_01_gates.py", "test_bg_q6_transport_retry_separate_from_attempt_lifecycle"),
            _nid("test_bg_01_gates.py", "test_bg_q6_transport_redelivery_does_not_reconcile_new_attempt"),
        ),
    ),
    Bg01QEvidence(
        "BG-Q7",
        "Terminal re-entry does not treat duplicate delivery as fresh execution",
        (_nid("test_bg_01_gates.py", "test_bg_q7_terminal_redelivery_safe_disposition"),),
    ),
    Bg01QEvidence(
        "BG-Q8",
        "Custom TaskQueue provider executes canonical host task and returns a valid canonical result without core changes",
        (_nid("test_bg_01_gates.py", "test_bg_q8_custom_task_queue_plugin"),),
    ),
    Bg01QEvidence(
        "BG-Q9",
        "Background intake contracts avoid concrete persistence vendor imports",
        (_nid("test_bg_01_gates.py", "test_bg_q9_background_intake_import_layer_gate"),),
    ),
    Bg01QEvidence(
        "BG-Q10",
        "Resume/checkpoint path re-enters host execution with resume_checkpoint",
        (
            _nid("test_bg_01_gates.py", "test_bg_q10_resume_path_uses_host_execution"),
            _nid("test_bg_01_gates.py", "test_bg_q10_resume_checkpoint_forwarded_to_host_execute"),
        ),
    ),
    Bg01QEvidence(
        "BG-Q11",
        "Worker runtime does not redefine execution timeout as transport lease",
        (_nid("test_bg_01_gates.py", "test_bg_q11_no_transport_lease_as_execution_timeout"),),
    ),
    Bg01QEvidence(
        "BG-Q12",
        "No legacy alternate execution adapters in production composition",
        (_nid("test_bg_01_gates.py", "test_bg_q12_no_legacy_alternate_execution_in_composition"),),
    ),
    Bg01QEvidence(
        "BG-Q13",
        "Canonical background contracts vendor-neutral",
        (_nid("test_bg_01_gates.py", "test_bg_q13_background_contracts_vendor_neutral"),),
    ),
    Bg01QEvidence(
        "BG-Q14",
        "Forbidden dynamic ABI patterns absent on background execution surface",
        (_nid("test_bg_01_gates.py", "test_bg_q14_forbidden_integration_patterns_gate"),),
    ),
    Bg01QEvidence(
        "BG-Q15",
        "Layer boundaries — worker transport does not reach model/tool/DB vendors",
        (_nid("test_bg_01_gates.py", "test_bg_q15_background_worker_layer_gate"),),
    ),
)
