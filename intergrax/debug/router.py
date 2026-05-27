# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Debug HTTP API (Phase D.2, architecture §19).

Endpoints mirror :mod:`intergrax.debug.cli`:

- ``GET /debug/tasks`` — list recent runs
- ``GET /debug/tasks/{run_id}`` — run metadata
- ``GET /debug/tasks/{run_id}/trace`` — trace timeline (+ optional runtime events)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Query

from intergrax.debug.formatters import build_trace_payload
from intergrax.debug.models import (
    RunDetailResponse,
    RunListResponse,
    RunSummaryItem,
    TraceResponse,
)
from intergrax.debug.store import open_trace_reader, resolve_trace_db_path
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader


def _trace_reader_factory(db_path: Path | None) -> Callable[[], RunTraceReader]:
    resolved = resolve_trace_db_path(str(db_path) if db_path is not None else None)

    def _open() -> RunTraceReader:
        try:
            return open_trace_reader(resolved)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _open


def create_debug_router(*, db_path: Path | None = None) -> APIRouter:
    router = APIRouter(prefix="/debug", tags=["debug"])
    get_reader = _trace_reader_factory(db_path)

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

    return router
