# © Artur Czarnecki. All rights reserved.

"""Read-only adaptive harness debug routes (Phase W-ADAPT-7.5)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from intergrax.runtime.adaptive.proposal_store import ProposalStore
from intergrax.runtime.adaptive.signal_store import SignalStore
from pydantic import BaseModel, Field


class AdaptiveSignalSummary(BaseModel):
    signal_id: str
    run_id: str
    tenant_id: str
    task_class: str
    utility: float | None = None
    eval_mode: str


class AdaptiveSignalListResponse(BaseModel):
    signals: list[AdaptiveSignalSummary] = Field(default_factory=list)
    count: int = 0


class AdaptiveProposalSummary(BaseModel):
    proposal_id: str
    loop_id: str
    source_engine: str
    passed_all_gates: bool


class AdaptiveProposalListResponse(BaseModel):
    proposals: list[AdaptiveProposalSummary] = Field(default_factory=list)
    count: int = 0


def create_adaptive_debug_router(
    *,
    signal_store: SignalStore | None,
    proposal_store: ProposalStore | None,
) -> APIRouter:
    """Create read-only adaptive debug endpoints for lab hosts."""
    router = APIRouter(prefix="/debug/adaptive", tags=["debug-adaptive"])

    @router.get("/signals", response_model=AdaptiveSignalListResponse)
    def list_adaptive_signals(
        tenant_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> AdaptiveSignalListResponse:
        if signal_store is None:
            raise HTTPException(status_code=404, detail="Adaptive signal store not configured")
        signals = signal_store.list_signals(tenant_id=tenant_id, limit=limit)
        summaries = [
            AdaptiveSignalSummary(
                signal_id=item.signal_id,
                run_id=item.run_id,
                tenant_id=item.tenant_id,
                task_class=item.task_class,
                utility=item.utility,
                eval_mode=item.eval_mode.value,
            )
            for item in signals
        ]
        return AdaptiveSignalListResponse(signals=summaries, count=len(summaries))

    @router.get("/proposals", response_model=AdaptiveProposalListResponse)
    def list_adaptive_proposals(
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> AdaptiveProposalListResponse:
        if proposal_store is None:
            raise HTTPException(status_code=404, detail="Adaptive proposal store not configured")
        runs = proposal_store.list_runs(limit=limit)
        summaries: list[AdaptiveProposalSummary] = []
        for run in runs:
            for package in run.packages:
                summaries.append(
                    AdaptiveProposalSummary(
                        proposal_id=package.proposal_id,
                        loop_id=package.candidate.loop_id,
                        source_engine=package.candidate.source_engine,
                        passed_all_gates=package.passed_all_gates,
                    )
                )
        return AdaptiveProposalListResponse(proposals=summaries, count=len(summaries))

    return router
