# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Debug HTTP API (Phase D.2–D.3, architecture §19, §35).

Task endpoints:

- ``GET /debug/tasks`` — list recent runs
- ``GET /debug/tasks/{run_id}`` — run metadata
- ``GET /debug/tasks/{run_id}/trace`` — trace timeline

Experiment registry:

- ``GET /debug/experiments`` — list experiments
- ``POST /debug/experiments`` — register experiment
- ``GET /debug/experiments/{experiment_id}`` — experiment details
- ``POST /debug/experiments/{experiment_id}/decision`` — keep/improve/pause/delete
- ``POST /debug/experiments/{experiment_id}/runs/{run_id}`` — link run
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from intergrax.debug.formatters import build_trace_payload
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.debug.models import (
    CheckpointItem,
    CheckpointListResponse,
    ExperimentDeletedResponse,
    ExperimentListResponse,
    HumanResponseResult,
    RunDetailResponse,
    RunListResponse,
    RunSummaryItem,
    RuntimeEventItem,
    RuntimeEventListResponse,
    SubmitHumanResponseRequest,
    TraceResponse,
)
from intergrax.debug.store import (
    open_default_task_checkpoint_persistence,
    open_runtime_event_persistence,
    open_task_checkpoint_persistence,
    open_trace_reader,
    resolve_trace_db_path,
)
from intergrax.experiments.models import (
    ExperimentDecision,
    ExperimentRecord,
    RegisterExperimentRequest,
    SetExperimentDecisionRequest,
)
from intergrax.experiments.store import (
    SQLiteExperimentStore,
    open_experiment_store,
    resolve_experiments_db_path,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointReader
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader


def _trace_reader_factory(db_path: Path | None) -> Callable[[], RunTraceReader]:
    resolved = resolve_trace_db_path(str(db_path) if db_path is not None else None)

    def _open() -> RunTraceReader:
        try:
            return open_trace_reader(resolved)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _open


def _experiment_store_factory(
    experiments_db_path: Path | None,
) -> Callable[[], SQLiteExperimentStore]:
    resolved = resolve_experiments_db_path(
        str(experiments_db_path) if experiments_db_path is not None else None
    )

    def _open() -> SQLiteExperimentStore:
        return open_experiment_store(resolved)

    return _open


def _runtime_event_store_factory(
    runtime_events_db_path: Path | None,
    implementation: RuntimeEventPersistence | None,
) -> Callable[[], RuntimeEventPersistence | None]:
    def _open() -> RuntimeEventPersistence | None:
        return open_runtime_event_persistence(
            db_path=runtime_events_db_path,
            implementation=implementation,
        )

    return _open


def _checkpoint_store_factory(
    checkpoints_db_path: Path | None,
    implementation: TaskCheckpointReader | None,
) -> Callable[[], TaskCheckpointReader | None]:
    def _open() -> TaskCheckpointReader | None:
        if implementation is not None:
            return implementation
        if checkpoints_db_path is not None:
            return open_task_checkpoint_persistence(db_path=checkpoints_db_path)
        return open_default_task_checkpoint_persistence()

    return _open


def _runtime_event_item(event) -> RuntimeEventItem:
    return RuntimeEventItem(
        event_id=event.event_id,
        event_type=event.event_type.value,
        task_id=event.task_id,
        run_id=event.run_id,
        tenant_id=event.tenant_id,
        phase=event.phase.value,
        severity=event.severity.value,
        timestamp=event.timestamp.isoformat(),
        payload=dict(event.payload),
    )


def create_debug_router(
    *,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    runtime_event_store: RuntimeEventPersistence | None = None,
    checkpoint_store: TaskCheckpointReader | None = None,
    hitl_service: DebugHitlResumeService | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/debug", tags=["debug"])
    get_reader = _trace_reader_factory(db_path)
    get_experiments = _experiment_store_factory(experiments_db_path)
    get_runtime_events = _runtime_event_store_factory(runtime_events_db_path, runtime_event_store)
    get_checkpoints = _checkpoint_store_factory(checkpoints_db_path, checkpoint_store)

    @router.get("/tasks", response_model=RunListResponse)
    def list_tasks(
        tenant: str = Query(default="default", description="Tenant id filter"),
        limit: int = Query(default=20, ge=1, le=200, description="Max rows"),
        reader: RunTraceReader = Depends(get_reader),
    ) -> RunListResponse:
        runs = reader.list_runs(tenant, limit=limit)
        return RunListResponse(
            tenant_id=tenant,
            count=len(runs),
            runs=[RunSummaryItem.from_summary(run) for run in runs],
        )

    @router.get("/tasks/{run_id}", response_model=RunDetailResponse)
    def show_task(
        run_id: str,
        tenant: str = Query(default="default", description="Tenant id"),
        reader: RunTraceReader = Depends(get_reader),
    ) -> RunDetailResponse:
        try:
            persisted = reader.read_run(run_id, tenant)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RunDetailResponse.from_persisted(persisted)

    @router.get("/tasks/{run_id}/trace", response_model=TraceResponse)
    def task_trace(
        run_id: str,
        tenant: str = Query(default="default", description="Tenant id"),
        include_runtime: bool = Query(
            default=False,
            description="Include RuntimeEvent view via trace_bridge",
        ),
        reader: RunTraceReader = Depends(get_reader),
    ) -> TraceResponse:
        try:
            persisted = reader.read_run(run_id, tenant)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        payload = build_trace_payload(persisted, include_runtime=include_runtime)
        return TraceResponse(
            run_id=str(payload["run_id"]),
            tenant_id=persisted.metadata.tenant_id,
            trace_events=list(payload["trace_events"]),
            runtime_events=payload.get("runtime_events"),
        )

    @router.get("/tasks/{task_id}/events", response_model=RuntimeEventListResponse)
    def task_runtime_events(
        task_id: str,
        tenant: str = Query(default="default", description="Tenant id"),
        limit: int = Query(default=200, ge=1, le=2000, description="Max events"),
        store: RuntimeEventPersistence | None = Depends(get_runtime_events),
    ) -> RuntimeEventListResponse:
        if store is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Runtime event persistence is not configured. "
                    "Pass runtime_event_store / runtime_events_db_path to create_debug_app "
                    "or set INTERGRAX_RUNTIME_EVENT_STORE."
                ),
            )
        events = store.list_for_task(task_id, tenant_id=tenant, limit=limit)
        if not events:
            events = store.list_for_run(task_id, tenant_id=tenant, limit=limit)
        return RuntimeEventListResponse(
            task_id=task_id,
            tenant_id=tenant,
            count=len(events),
            events=[_runtime_event_item(event) for event in events],
        )

    @router.get("/tasks/{task_id}/checkpoints", response_model=CheckpointListResponse)
    def task_checkpoints(
        task_id: str,
        tenant: str = Query(default="default", description="Tenant id"),
        reader: TaskCheckpointReader | None = Depends(get_checkpoints),
    ) -> CheckpointListResponse:
        if reader is None:
            raise HTTPException(
                status_code=503,
                detail="Checkpoint persistence is not configured.",
            )
        rows = reader.list_for_task(task_id, tenant)
        return CheckpointListResponse(
            task_id=task_id,
            tenant_id=tenant,
            count=len(rows),
            checkpoints=[
                CheckpointItem(
                    checkpoint_id=row.checkpoint_id,
                    task_id=row.task_id,
                    tenant_id=row.tenant_id,
                    resume_token=row.resume_token,
                    task_state=row.task_state.value,
                    progress_message=row.progress_message,
                    notify_channel=row.notify_channel,
                    created_at_utc=row.created_at_utc,
                    has_runtime_checkpoint=row.runtime is not None,
                )
                for row in rows
            ],
        )

    @router.post("/tasks/{task_id}/human-response", response_model=HumanResponseResult)
    async def submit_human_response(
        task_id: str,
        body: SubmitHumanResponseRequest,
        tenant: str = Query(default="default", description="Tenant id"),
    ) -> HumanResponseResult:
        if hitl_service is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "HITL resume is not configured. "
                    "Pass registry=AgentRegistry(...) to create_debug_app."
                ),
            )
        try:
            result = await hitl_service.resume_with_human_response(
                task_id,
                tenant,
                response=body.response,
                resume_token=body.resume_token,
                user_id=body.user_id or "debug_operator",
            )
        except ValueError as exc:
            message = str(exc)
            if message.startswith("No checkpoint found"):
                raise HTTPException(status_code=404, detail=message) from exc
            raise HTTPException(status_code=409, detail=message) from exc

        return HumanResponseResult(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            resume_token=result.summary.resume_token,
            checkpoint_id=result.summary.checkpoint_id,
        )

    @router.get("/experiments", response_model=ExperimentListResponse)
    def list_experiments(
        limit: int = Query(default=20, ge=1, le=200),
        decision: ExperimentDecision | None = Query(default=None),
        store: SQLiteExperimentStore = Depends(get_experiments),
    ) -> ExperimentListResponse:
        records = store.list_experiments(limit=limit, decision=decision)
        return ExperimentListResponse(
            count=len(records),
            experiments=[record.model_dump(mode="json") for record in records],
        )

    @router.post("/experiments", response_model=ExperimentRecord, status_code=201)
    def register_experiment(
        body: RegisterExperimentRequest,
        store: SQLiteExperimentStore = Depends(get_experiments),
    ) -> ExperimentRecord:
        return store.register(body)

    @router.get("/experiments/{experiment_id}", response_model=ExperimentRecord)
    def show_experiment(
        experiment_id: str,
        store: SQLiteExperimentStore = Depends(get_experiments),
    ) -> ExperimentRecord:
        try:
            return store.get(experiment_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/experiments/{experiment_id}/decision")
    def decide_experiment(
        experiment_id: str,
        body: SetExperimentDecisionRequest,
        store: SQLiteExperimentStore = Depends(get_experiments),
    ) -> ExperimentRecord | ExperimentDeletedResponse:
        try:
            record = store.set_decision(
                experiment_id,
                body.decision,
                notes=body.notes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if body.decision == ExperimentDecision.DELETE:
            return ExperimentDeletedResponse(experiment_id=experiment_id)
        return record

    @router.post("/experiments/{experiment_id}/runs/{run_id}", response_model=ExperimentRecord)
    def link_experiment_run(
        experiment_id: str,
        run_id: str,
        store: SQLiteExperimentStore = Depends(get_experiments),
    ) -> ExperimentRecord:
        try:
            return store.link_run(experiment_id, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
