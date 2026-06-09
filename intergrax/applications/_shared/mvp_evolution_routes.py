# © Artur Czarnecki. All rights reserved.

"""HTTP exposure for MVP evolution CLI (MVP-EVOL.7)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from intergrax.cli.mvp_evolution import run_mvp_replay, run_mvp_simulate


def create_mvp_evolution_router(*, enabled: bool = True) -> APIRouter:
    router = APIRouter(prefix="/mvp", tags=["mvp-evolution"])

    @router.post("/simulate")
    def mvp_simulate() -> dict[str, str]:
        if not enabled:
            raise HTTPException(status_code=404, detail="mvp routes disabled")
        run_mvp_simulate()
        return {"status": "ok", "action": "simulate"}

    @router.post("/replay")
    def mvp_replay() -> dict[str, str]:
        if not enabled:
            raise HTTPException(status_code=404, detail="mvp routes disabled")
        run_mvp_replay()
        return {"status": "ok", "action": "replay"}

    return router
