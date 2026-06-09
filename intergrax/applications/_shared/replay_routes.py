# © Artur Czarnecki. All rights reserved.

"""Harness replay HTTP API for product hosts (AUDIT-IDEAL-27.2)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from intergrax.cli.mvp_evolution import run_mvp_replay


class ReplayRequest(BaseModel):
    run_id: str = Field(min_length=1)
    trace_id: str | None = None


def create_replay_router(*, enabled: bool = True) -> APIRouter:
    router = APIRouter(prefix="/harness", tags=["harness-replay"])

    @router.post("/replay")
    def harness_replay(body: ReplayRequest) -> dict[str, str]:
        if not enabled:
            raise HTTPException(status_code=404, detail="replay routes disabled")
        run_mvp_replay()
        return {
            "status": "ok",
            "action": "replay",
            "run_id": body.run_id,
            "trace_id": body.trace_id or "",
        }

    return router
