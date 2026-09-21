# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X4 — real cross-process Kafka async and HITL recovery spine E2E."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from intergrax.integrations.providers.message_bus.kafka.config import DEFAULT_BOOTSTRAP_SERVERS
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.task.task import TaskState
from testing_support.cross_process_spine.harness import CrossProcessSpineHarness
from testing_support.cross_process_spine.kafka_probe import kafka_broker_ready

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.obs_coverage_p4,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KAFKA_BOOTSTRAP = os.environ.get(
    "INTERGRAX_KAFKA_BOOTSTRAP_SERVERS",
    DEFAULT_BOOTSTRAP_SERVERS,
).strip()


def _kafka_available() -> bool:
    try:
        kafka_broker_ready(_KAFKA_BOOTSTRAP, timeout_seconds=8.0)
        return True
    except (TimeoutError, OSError):
        return False


@pytest.fixture
def x4_harness(tmp_path: Path) -> CrossProcessSpineHarness:
    harness = CrossProcessSpineHarness(repo_root=_REPO_ROOT, work_dir=tmp_path / "x4")
    yield harness
    harness.cleanup()


@pytest.mark.skipif(not _kafka_available(), reason="real Kafka broker unavailable")
def test_kafka_cross_process_success_no_false_problem(x4_harness: CrossProcessSpineHarness) -> None:
    worker, fresh, _transport_task_id = x4_harness.run_kafka_worker_proof(
        bootstrap_servers=_KAFKA_BOOTSTRAP,
        inject_violation=False,
    )
    assert worker.worker_pid != fresh.reader_pid
    assert worker.status == "succeeded"
    assert worker.terminal_event_type == RuntimeEventType.TASK_COMPLETED.value
    assert worker.problem_count == 0
    assert fresh.problem_count == 0
    assert fresh.reconstruction_has_events


@pytest.mark.skipif(not _kafka_available(), reason="real Kafka broker unavailable")
def test_kafka_cross_process_failure_creates_durable_problem(
    x4_harness: CrossProcessSpineHarness,
) -> None:
    worker, fresh, _ = x4_harness.run_kafka_worker_proof(
        bootstrap_servers=_KAFKA_BOOTSTRAP,
        inject_violation=True,
    )
    assert worker.problem_count == 1
    assert fresh.problem_count == 1
    assert worker.problem_ids == fresh.problem_ids


@pytest.mark.skipif(not _kafka_available(), reason="real Kafka broker unavailable")
def test_kafka_cross_process_identity_lineage(x4_harness: CrossProcessSpineHarness) -> None:
    worker, fresh, transport_task_id = x4_harness.run_kafka_worker_proof(
        bootstrap_servers=_KAFKA_BOOTSTRAP,
        inject_violation=False,
    )
    assert worker.transport_task_id == transport_task_id
    assert worker.run_id
    assert worker.task_id
    assert worker.execution_id
    assert fresh.reconstruction_has_events


@pytest.mark.skipif(not _kafka_available(), reason="real Kafka broker unavailable")
def test_kafka_cross_process_fresh_reader(x4_harness: CrossProcessSpineHarness) -> None:
    worker, fresh, _ = x4_harness.run_kafka_worker_proof(
        bootstrap_servers=_KAFKA_BOOTSTRAP,
        inject_violation=True,
    )
    assert fresh.problem_count == worker.problem_count
    assert fresh.reconstruction_has_events


@pytest.mark.skipif(not _kafka_available(), reason="real Kafka broker unavailable")
def test_kafka_duplicate_delivery_idempotent(x4_harness: CrossProcessSpineHarness) -> None:
    worker, fresh, _ = x4_harness.run_kafka_worker_proof(
        bootstrap_servers=_KAFKA_BOOTSTRAP,
        inject_violation=False,
        idempotency_key="x4-dup-delivery",
        duplicate_delivery=True,
    )
    assert worker.handler_invocation_count == 1
    assert fresh.problem_count == 0


def test_kafka_worker_enters_execute_logical_task_not_direct_agent() -> None:
    broker_source = (
        _REPO_ROOT / "intergrax" / "queueing" / "providers" / "broker_worker_base.py"
    ).read_text(encoding="utf-8")
    assert "execute_logical_task" in broker_source
    assert "admit_background_execution_handler" in broker_source
    worker_source = (
        _REPO_ROOT / "intergrax" / "queueing" / "providers" / "kafka" / "kafka_worker.py"
    ).read_text(encoding="utf-8")
    assert "DiagnosticOrchestrator" not in worker_source


def test_hitl_cross_process_new_pid_pause_and_resume(x4_harness: CrossProcessSpineHarness) -> None:
    pause, resume, fresh = x4_harness.run_hitl_cross_process_proof(human_approved=True)
    assert pause.phase == "pause"
    assert pause.problem_count == 0
    assert pause.process_pid != os.getpid()
    assert resume.process_pid != pause.process_pid
    assert resume.terminal_state == TaskState.COMPLETED.value
    assert fresh.problem_count == 0
    assert fresh.reconstruction_has_events


def test_hitl_cross_process_no_false_problem_on_valid_pause(x4_harness: CrossProcessSpineHarness) -> None:
    pause, _, fresh = x4_harness.run_hitl_cross_process_proof(human_approved=True)
    assert pause.pause_state == TaskState.WAITING_FOR_HUMAN.value
    assert pause.problem_count == 0
    assert fresh.problem_count == 0


def test_hitl_cross_process_resume_success(x4_harness: CrossProcessSpineHarness) -> None:
    pause, resume, _ = x4_harness.run_hitl_cross_process_proof(human_approved=True)
    assert pause.task_id == resume.task_id
    assert pause.run_id == resume.run_id
    assert resume.terminal_state == TaskState.COMPLETED.value


def test_hitl_cross_process_human_rejection_terminal_behavior(x4_harness: CrossProcessSpineHarness) -> None:
    pause, resume, fresh = x4_harness.run_hitl_cross_process_proof(
        human_approved=False,
        human_rejected=True,
    )
    assert pause.task_id == resume.task_id
    assert pause.run_id == resume.run_id
    assert pause.execution_id == resume.execution_id
    assert resume.terminal_state == TaskState.FAILED.value
    assert resume.terminal_event_type == RuntimeEventType.TASK_FAILED.value
    assert resume.problem_count == 0
    assert fresh.problem_count == 0
    assert fresh.reconstruction_has_events


def test_hitl_cross_process_resume_injected_violation_creates_durable_problem(
    x4_harness: CrossProcessSpineHarness,
) -> None:
    pause, resume, fresh = x4_harness.run_hitl_cross_process_proof(
        human_approved=True,
        inject_violation_on_resume=True,
    )
    assert pause.process_pid != resume.process_pid
    assert pause.task_id == resume.task_id
    assert pause.run_id == resume.run_id
    assert pause.attempt_id == resume.attempt_id
    assert pause.execution_id == resume.execution_id
    assert resume.terminal_state == TaskState.COMPLETED.value
    assert resume.terminal_event_type == RuntimeEventType.TASK_COMPLETED.value
    assert resume.problem_count >= 1
    assert fresh.problem_count >= 1
    assert resume.problem_ids == fresh.problem_ids
    assert fresh.reconstruction_has_events
    assert fresh.reconstruction_complete


def test_hitl_cross_process_fresh_read_and_reconstruction(x4_harness: CrossProcessSpineHarness) -> None:
    pause, resume, fresh = x4_harness.run_hitl_cross_process_proof(human_approved=True)
    assert pause.run_id == resume.run_id
    assert fresh.reconstruction_has_events
    assert fresh.reconstruction_complete
