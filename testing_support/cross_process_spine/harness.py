# © Artur Czarnecki. All rights reserved.

"""Parent-process driver for OBS-DIAG-X4 cross-process spine proofs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.bundle import resolve_kafka_config
from intergrax.integrations.providers.message_bus.kafka.opens import (
    open_kafka_producer,
    open_kafka_task_queue,
)
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from testing_support.cross_process_spine.kafka_probe import ensure_kafka_topic, kafka_broker_ready
from testing_support.cross_process_spine.models import (
    CrossProcessFreshReadResult,
    HitlProcessResult,
    HitlSpineScenarioConfig,
    KafkaSpineScenarioConfig,
    KafkaWorkerResult,
)
from testing_support.cross_process_spine.obs_spine_kafka_stack import (
    _OBS_SPINE_TASK,
    fresh_diagnostic_read_from_durable_backing,
)


@dataclass(frozen=True, slots=True)
class SubprocessRun:
    completed: subprocess.CompletedProcess[str]
    result_path: Path


class CrossProcessSpineHarness:
    """Broker/worker lifecycle + bounded subprocess orchestration for X4 proofs."""

    def __init__(self, *, repo_root: Path, work_dir: Path) -> None:
        self._repo_root = repo_root
        self._work_dir = work_dir
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._child_processes: list[subprocess.Popen[object]] = []

    def cleanup(self) -> None:
        for proc in self._child_processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._child_processes.clear()

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        root = str(self._repo_root)
        agents = str(self._repo_root / "agents")
        applications = str(self._repo_root / "applications")
        separator = ";" if sys.platform.startswith("win") else ":"
        extra = separator.join([root, agents, applications])
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = f"{extra}{separator}{existing}" if existing else extra
        return env

    def _run_module(
        self,
        module: str,
        args: list[str],
        *,
        timeout_seconds: float = 180.0,
    ) -> SubprocessRun:
        completed = subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=str(self._repo_root),
            env=self._child_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return SubprocessRun(completed=completed, result_path=Path(args[-1]))

    def require_kafka(self, bootstrap_servers: str) -> None:
        kafka_broker_ready(bootstrap_servers)

    def run_kafka_worker_proof(
        self,
        *,
        bootstrap_servers: str,
        inject_violation: bool,
        idempotency_key: str | None = None,
        duplicate_delivery: bool = False,
        tenant_id: str = "tenant-obs-spine-kafka-x4",
    ) -> tuple[KafkaWorkerResult, CrossProcessFreshReadResult, int]:
        self.require_kafka(bootstrap_servers)
        scenario_id = f"x4-kafka-{uuid.uuid4().hex[:10]}"
        work_dir = self._work_dir / scenario_id
        work_dir.mkdir(parents=True, exist_ok=True)
        topic = f"intergrax-x4-{uuid.uuid4().hex}"
        group_id = f"intergrax-x4-group-{uuid.uuid4().hex}"
        run_id = str(mint_run_id())
        config = KafkaSpineScenarioConfig(
            scenario_id=scenario_id,
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            consumer_group=group_id,
            tenant_id=tenant_id,
            run_id=run_id,
            inject_violation=inject_violation,
            idempotency_key=idempotency_key,
            duplicate_delivery=duplicate_delivery,
            work_dir=str(work_dir),
            kv_db_path=str(work_dir / "worker_kv.sqlite"),
            runtime_events_db_path=str(work_dir / "runtime_events.sqlite"),
            document_store_db_path=str(work_dir / "problems.docstore.sqlite"),
            causal_evidence_db_path=str(work_dir / "causal.sqlite"),
            max_messages=2 if duplicate_delivery else 1,
        )
        config_path = work_dir / "kafka_config.json"
        result_path = work_dir / "kafka_worker_result.json"
        config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

        ensure_kafka_topic(bootstrap_servers, topic)

        worker_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "testing_support.cross_process_spine.kafka_worker_cli",
                str(config_path),
                str(result_path),
            ],
            cwd=str(self._repo_root),
            env=self._child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._child_processes.append(worker_proc)

        ready_path = work_dir / "worker_ready.json"
        ready_deadline = time.monotonic() + 30.0
        while time.monotonic() < ready_deadline:
            if ready_path.is_file():
                break
            if worker_proc.poll() is not None:
                stdout, stderr = worker_proc.communicate(timeout=1)
                raise RuntimeError(
                    f"Kafka worker failed during startup code={worker_proc.returncode} "
                    f"stdout={stdout} stderr={stderr}",
                )
            time.sleep(0.1)
        else:
            raise TimeoutError("Kafka worker subprocess did not signal readiness")

        kv_store = SqliteDistributedKVStore(work_dir / "worker_kv.sqlite")
        kafka_config = resolve_kafka_config(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            consumer_group=group_id,
        )
        producer = open_kafka_producer(kafka_config)
        task_queue = open_kafka_task_queue(
            kafka_config,
            kv_store=kv_store,
            topic=topic,
            producer=producer,
        )
        request = TaskRequest(
            tenant_id=tenant_id,
            run_id=run_id,
            task_name=_OBS_SPINE_TASK,
            payload=b"{}",
            idempotency_key=idempotency_key,
        )
        handle = task_queue.enqueue(request=request)
        if duplicate_delivery and idempotency_key is not None:
            task_queue.enqueue(
                request=TaskRequest(
                    tenant_id=tenant_id,
                    run_id=run_id,
                    task_name=_OBS_SPINE_TASK,
                    payload=b"{}",
                    idempotency_key=idempotency_key,
                ),
            )

        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            if result_path.is_file():
                break
            if worker_proc.poll() is not None and not result_path.is_file():
                stdout, stderr = worker_proc.communicate(timeout=1)
                raise RuntimeError(
                    f"Kafka worker exited early code={worker_proc.returncode} "
                    f"stdout={stdout} stderr={stderr}",
                )
            time.sleep(0.2)
        else:
            worker_proc.terminate()
            raise TimeoutError("Kafka worker subprocess timed out waiting for result artifact")

        stdout, stderr = worker_proc.communicate(timeout=30)
        if worker_proc.returncode not in (0, None):
            raise RuntimeError(
                f"Kafka worker failed code={worker_proc.returncode} stderr={stderr} stdout={stdout}",
            )

        worker_result = KafkaWorkerResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        assert worker_result.worker_pid != os.getpid()

        status_bytes = kv_store.get(tenant_id, f"task:{handle.task_id}:status")
        assert status_bytes is not None
        assert status_bytes.decode("utf-8") in {
            TaskStatus.SUCCEEDED.value,
            TaskStatus.FAILED.value,
        }

        problem_count, problem_ids, has_events, complete = fresh_diagnostic_read_from_durable_backing(
            work_dir=work_dir,
            tenant_id=tenant_id,
            task_id=worker_result.task_id,
            run_id=worker_result.run_id,
        )
        fresh_read = CrossProcessFreshReadResult(
            reader_pid=os.getpid(),
            problem_count=problem_count,
            problem_ids=problem_ids,
            reconstruction_has_events=has_events,
            reconstruction_complete=complete,
        )
        return worker_result, fresh_read, handle.task_id

    def run_hitl_cross_process_proof(
        self,
        *,
        human_approved: bool,
        human_rejected: bool = False,
        inject_violation_on_resume: bool = False,
    ) -> tuple[HitlProcessResult, HitlProcessResult, CrossProcessFreshReadResult]:
        scenario_id = f"x4-hitl-{uuid.uuid4().hex[:10]}"
        work_dir = self._work_dir / scenario_id
        work_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(mint_run_id())
        tenant_id = "tenant-obs-spine-hitl-x4"
        base = dict(
            scenario_id=scenario_id,
            tenant_id=tenant_id,
            run_id=run_id,
            work_dir=str(work_dir),
            checkpoint_db_path=str(work_dir / "checkpoints.sqlite"),
            runtime_events_db_path=str(work_dir / "runtime_events.sqlite"),
            document_store_db_path=str(work_dir / "problems.docstore.sqlite"),
            continuation_export_path=str(work_dir / "continuation_export.json"),
        )
        pause_config_path = work_dir / "hitl_pause_config.json"
        pause_result_path = work_dir / "hitl_pause_result.json"
        pause_config = HitlSpineScenarioConfig(phase="pause", **base)
        pause_config_path.write_text(pause_config.model_dump_json(indent=2), encoding="utf-8")
        pause_run = self._run_module(
            "testing_support.cross_process_spine.hitl_worker_cli",
            [str(pause_config_path), str(pause_result_path)],
        )
        if pause_run.completed.returncode != 0:
            raise RuntimeError(
                f"HITL pause subprocess failed: {pause_run.completed.stderr} {pause_run.completed.stdout}",
            )
        pause_result = HitlProcessResult.model_validate_json(
            pause_result_path.read_text(encoding="utf-8"),
        )
        assert pause_result.process_pid != os.getpid()

        resume_config_path = work_dir / "hitl_resume_config.json"
        resume_result_path = work_dir / "hitl_resume_result.json"
        resume_config = HitlSpineScenarioConfig(
            phase="resume",
            human_approved=human_approved,
            human_rejected=human_rejected,
            inject_violation_on_resume=inject_violation_on_resume,
            task_id=pause_result.task_id,
            resume_token=pause_result.resume_token,
            **base,
        )
        resume_config_path.write_text(resume_config.model_dump_json(indent=2), encoding="utf-8")
        resume_run = self._run_module(
            "testing_support.cross_process_spine.hitl_worker_cli",
            [str(resume_config_path), str(resume_result_path)],
        )
        if resume_run.completed.returncode != 0:
            raise RuntimeError(
                f"HITL resume subprocess failed: {resume_run.completed.stderr} {resume_run.completed.stdout}",
            )
        resume_result = HitlProcessResult.model_validate_json(
            resume_result_path.read_text(encoding="utf-8"),
        )
        assert resume_result.process_pid != os.getpid()
        assert resume_result.process_pid != pause_result.process_pid

        problem_count, problem_ids, has_events, complete = fresh_diagnostic_read_from_durable_backing(
            work_dir=work_dir,
            tenant_id=tenant_id,
            task_id=resume_result.task_id,
            run_id=run_id,
        )
        fresh_read = CrossProcessFreshReadResult(
            reader_pid=os.getpid(),
            problem_count=problem_count,
            problem_ids=problem_ids,
            reconstruction_has_events=has_events,
            reconstruction_complete=complete,
        )
        return pause_result, resume_result, fresh_read
