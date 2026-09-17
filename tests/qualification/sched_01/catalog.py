# © Artur Czarnecki. All rights reserved.

"""SCHED-01 Q1..Q15 evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Sched01QEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]


def _nid(test_file: str, test_name: str) -> str:
    return f"tests/qualification/sched_01/{test_file}::{test_name}"


def _pcm(test_name: str) -> str:
    return f"tests/unit/runtime/long_running/test_pcm_scheduler_integrity.py::{test_name}"


SCHED_01_Q_CATALOG: tuple[Sched01QEvidence, ...] = (
    Sched01QEvidence(
        "SCHED-Q1",
        "Due occurrence reaches HostTaskExecutionPort via scheduler resume executor",
        (_nid("test_sched_01_gates.py", "test_sched_q1_due_occurrence_invokes_host_execution_port"),),
    ),
    Sched01QEvidence(
        "SCHED-Q2",
        "Schedule definition durable through ScheduledResumePersistence abstraction",
        (_nid("test_sched_01_gates.py", "test_sched_q2_schedule_survives_store_reopen"),),
    ),
    Sched01QEvidence(
        "SCHED-Q3",
        "Tenant mismatch fail-closed before resume dispatch",
        (_nid("test_sched_01_gates.py", "test_sched_q3_tenant_mismatch_blocks_resume"),),
    ),
    Sched01QEvidence(
        "SCHED-Q4",
        "Stable schedule_id for one logical scheduled occurrence",
        (_nid("test_sched_01_gates.py", "test_sched_q4_schedule_id_stable_across_reads"),),
    ),
    Sched01QEvidence(
        "SCHED-Q5",
        "Duplicate due evaluation yields one claim and one resume",
        (
            _pcm("test_atomic_due_claim_exactly_one_winner"),
            _pcm("test_two_schedulers_single_resume_call"),
        ),
    ),
    Sched01QEvidence(
        "SCHED-Q6",
        "Atomic claim / fenced completion",
        (
            _pcm("test_fenced_completion_rejected"),
            _pcm("test_active_claim_blocks_second_owner"),
        ),
    ),
    Sched01QEvidence(
        "SCHED-Q7",
        "Overdue one-shot fires on poll; not before due",
        (_nid("test_sched_01_gates.py", "test_sched_q7_misfire_late_due_fires_once"),),
    ),
    Sched01QEvidence(
        "SCHED-Q8",
        "Scheduler polling retry does not mint new execution identity",
        (_nid("test_sched_01_gates.py", "test_sched_q8_scheduler_retry_preserves_checkpoint_identity"),),
    ),
    Sched01QEvidence(
        "SCHED-Q9",
        "Uncertain claim after lease expiry prevents duplicate logical resume",
        (_pcm("test_expired_running_becomes_uncertain_no_second_resume"),),
    ),
    Sched01QEvidence(
        "SCHED-Q10",
        "Custom in-memory schedule store via public persistence contract",
        (_nid("test_sched_01_gates.py", "test_sched_q10_custom_schedule_store_provider"),),
    ),
    Sched01QEvidence(
        "SCHED-Q11",
        "Explicit tick clock: before due no dispatch; at due dispatch",
        (_nid("test_sched_01_gates.py", "test_sched_q11_fake_clock_boundaries"),),
    ),
    Sched01QEvidence(
        "SCHED-Q12",
        "Cancelled pending schedule not dispatched; active claim blocks cancel",
        (
            _nid("test_sched_01_gates.py", "test_sched_q12_cancelled_pending_not_dispatched"),
            _pcm("test_cancel_active_running_rejected"),
        ),
    ),
    Sched01QEvidence(
        "SCHED-Q13",
        "Production scheduler wiring behaviorally resumes through HostTaskExecutionPort",
        (
            _nid(
                "test_sched_01_gates.py",
                "test_sched_q13_production_wiring_resumes_through_host_task_execution_port",
            ),
        ),
    ),
    Sched01QEvidence(
        "SCHED-Q14",
        "Scheduling core free of vendor persistence / broker imports",
        (_nid("test_sched_01_gates.py", "test_sched_q14_scheduling_core_import_layer_gate"),),
    ),
    Sched01QEvidence(
        "SCHED-Q15",
        "Scheduler core does not instantiate alternate execution runtime",
        (_nid("test_sched_01_gates.py", "test_sched_q15_no_alternate_execution_engine_in_scheduler_core"),),
    ),
)
