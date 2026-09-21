# © Artur Czarnecki. All rights reserved.

"""Subprocess Kafka worker entry for OBS-DIAG-X4."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from confluent_kafka import KafkaException

from intergrax.contracts.execution_identity import RunId
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_worker
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.runtime.diagnostics.persistence_conformance import query_all_problems_for_tenant
from testing_support.cross_process_spine.models import KafkaSpineScenarioConfig, KafkaWorkerResult
from testing_support.cross_process_spine.obs_spine_kafka_stack import (
    build_obs_spine_kafka_worker_stack,
    terminal_event_type_for_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OBS-DIAG-X4 Kafka worker subprocess")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args(argv)

    config = KafkaSpineScenarioConfig.model_validate_json(
        args.config_path.read_text(encoding="utf-8"),
    )
    work_dir = Path(config.work_dir)
    stack = build_obs_spine_kafka_worker_stack(
        work_dir=work_dir,
        inject_violation=config.inject_violation,
    )
    ready_path = work_dir / "worker_ready.json"
    ready_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    worker = create_kafka_worker(
        kv_store=stack.kv_store,
        execution_registry=stack.registry,
        idempotency_store=stack.idempotency_store,
        bootstrap_servers=config.bootstrap_servers,
        topic=config.topic,
        consumer_group=config.consumer_group,
        poll_timeout_seconds=config.poll_timeout_seconds,
        causal_evidence_persistence=stack.causal_store,
    )

    processed = 0
    deadline = time.monotonic() + 60.0
    last_task_id: str | None = None
    while processed < config.max_messages and time.monotonic() < deadline:
        try:
            raw = worker._consumer.poll(timeout_seconds=config.poll_timeout_seconds)  # noqa: SLF001
        except KafkaException:
            continue
        if raw is None:
            continue
        worker.process_message(raw_payload=raw)
        processed += 1
        try:
            message = json.loads(raw.decode("utf-8"))
            last_task_id = str(message.get("task_id", ""))
        except Exception:
            last_task_id = None

    if last_task_id is None:
        raise RuntimeError("kafka worker subprocess did not process any message")

    status_bytes = stack.kv_store.get(config.tenant_id, f"task:{last_task_id}:status")
    if status_bytes is None:
        raise RuntimeError("kafka worker missing terminal task status in durable KV")
    status = status_bytes.decode("utf-8")
    task_id = stack.completed_task_ids[-1] if stack.completed_task_ids else ""
    events = (
        stack.runtime_event_store.list_for_task(task_id, tenant_id=config.tenant_id)
        if task_id
        else stack.runtime_event_store.list_for_run(RunId(config.run_id), tenant_id=config.tenant_id)
    )
    if not task_id and events:
        task_id = str(events[0].task_id)
    resolved_run_id = str(events[0].run_id) if events else config.run_id
    execution_id = events[0].execution_id if events else ""
    attempt_id = events[0].attempt_id if events else ""
    terminal = terminal_event_type_for_run(
        runtime_store=stack.runtime_event_store,
        tenant_id=config.tenant_id,
        run_id=RunId(resolved_run_id),
    )
    problems = query_all_problems_for_tenant(stack.read_deps.problem_persistence, config.tenant_id)

    result = KafkaWorkerResult(
        worker_pid=os.getpid(),
        parent_pid=os.getppid(),
        tenant_id=config.tenant_id,
        run_id=resolved_run_id,
        task_id=str(task_id),
        transport_task_id=last_task_id,
        execution_id=str(execution_id),
        attempt_id=str(attempt_id),
        terminal_event_type=terminal.value if terminal is not None else "",
        problem_count=len(problems),
        problem_ids=tuple(problem.problem_id for problem in problems),
        handler_invocation_count=stack.handler_invocation_counter[0],
        status="succeeded" if status == TaskStatus.SUCCEEDED.value else "failed",
    )
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
