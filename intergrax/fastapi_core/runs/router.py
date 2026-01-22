# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from intergrax.fastapi_core import rate_limit
from intergrax.fastapi_core.auth.dependency import require_scope
from intergrax.fastapi_core.rate_limit.keys import RateLimitKey
from intergrax.fastapi_core.runs.models import CreateRunRequest, RunResponse
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore

runs_router = APIRouter(prefix="/runs", tags=["runs"])


@runs_router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED,)
def create_run(
    store: RunStore = Depends(),
    _: None = Depends(rate_limit(RateLimitKey.TENANT)),
    __=Depends(require_scope("runs:create")),
    ___: CreateRunRequest | None = None,
) -> RunResponse:
    return store.create()


@runs_router.get("/{run_id}", response_model=RunResponse,)
def get_run(    
    run_id: str,
    store: RunStore = Depends(),
    __=Depends(require_scope("runs:read")),
) -> RunResponse:
    try:
        return store.get(run_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )


@runs_router.post("/{run_id}/cancel",response_model=RunResponse,)
def cancel_run(
    run_id: str,
    store: RunStore = Depends(),
    __=Depends(require_scope("runs:cancel")),
) -> RunResponse:
    try:
        return store.cancel(run_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )
