# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, status
from intergrax.fastapi_core.budget.dependency import require_budget
from intergrax.fastapi_core.context import RequestContext, get_request_context
from intergrax.fastapi_core.rate_limit.dependency import rate_limit
from intergrax.fastapi_core.auth.dependency import require_scope
from intergrax.fastapi_core.rate_limit.keys import RateLimitKey
from intergrax.fastapi_core.runs.models import CreateRunRequest, RunResponse
from intergrax.fastapi_core.runs.service import RunService

runs_router = APIRouter(prefix="/runs", tags=["runs"])


@runs_router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED,)
def create_run(
    background_tasks: BackgroundTasks,
    request: CreateRunRequest = Body(...),  
    context: RequestContext = Depends(get_request_context),  
    service: RunService = Depends(),    
    _: None = Depends(rate_limit(RateLimitKey.TENANT)),
    __=Depends(require_scope("runs:create")),
    ___=Depends(require_budget()),
) -> RunResponse:
    return service.create_run(context, background_tasks)



@runs_router.get("/{run_id}", response_model=RunResponse,)
def get_run(
    run_id: str,
    service: RunService = Depends(),
    __=Depends(require_scope("runs:read")),
) -> RunResponse:
    try:
        return service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")



@runs_router.post("/{run_id}/cancel",response_model=RunResponse,)
def cancel_run(
    run_id: str,
    service: RunService = Depends(),
    __=Depends(require_scope("runs:cancel")),
) -> RunResponse:
    try:
        return service.cancel_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
