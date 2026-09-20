# © Artur Czarnecki. All rights reserved.

"""Typed serializable configs for OBS-DIAG-X4 cross-process proofs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class KafkaSpineScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    bootstrap_servers: str = "localhost:9092"
    topic: str
    consumer_group: str
    tenant_id: str
    run_id: str
    task_name: str = "obs_spine.cross_process.kafka.v1"
    inject_violation: bool = False
    idempotency_key: str | None = None
    duplicate_delivery: bool = False
    work_dir: str
    kv_db_path: str
    runtime_events_db_path: str
    document_store_db_path: str
    causal_evidence_db_path: str
    max_messages: int = 1
    poll_timeout_seconds: float = 0.5


class KafkaWorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_pid: int
    parent_pid: int
    tenant_id: str
    run_id: str
    task_id: str
    transport_task_id: str
    execution_id: str
    attempt_id: str
    terminal_event_type: str
    problem_count: int
    problem_ids: tuple[str, ...] = ()
    handler_invocation_count: int = 1
    status: Literal["succeeded", "failed"]


class HitlSpineScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    tenant_id: str = "tenant-obs-spine-hitl-x4"
    run_id: str
    work_dir: str
    checkpoint_db_path: str
    runtime_events_db_path: str
    document_store_db_path: str
    continuation_export_path: str
    human_approved: bool = True
    human_rejected: bool = False
    inject_violation_on_resume: bool = False
    phase: Literal["pause", "resume"]
    task_id: str | None = None
    resume_token: str | None = None


class HitlProcessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_pid: int
    parent_pid: int
    phase: Literal["pause", "resume"]
    tenant_id: str
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str
    terminal_state: str | None = None
    terminal_event_type: str | None = None
    pause_state: str | None = None
    resume_token: str | None = None
    problem_count: int = 0
    problem_ids: tuple[str, ...] = ()
    reconstruction_has_events: bool = False
    reconstruction_complete: bool = False


class CrossProcessFreshReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reader_pid: int
    problem_count: int
    problem_ids: tuple[str, ...] = ()
    reconstruction_has_events: bool
    reconstruction_complete: bool
