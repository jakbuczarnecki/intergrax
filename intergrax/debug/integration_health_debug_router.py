# © Artur Czarnecki. All rights reserved.

"""Read-only integration catalog health probes for lab/debug hosts (Phase W-OPS.10)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from intergrax.integrations.registry.harness_lab_health import (
    health_check_harness_lab_stack,
    health_check_harness_m6_p4_probes,
)


class IntegrationHealthItem(BaseModel):
    slug: str
    healthy: bool
    detail: str = ""


class IntegrationHealthResponse(BaseModel):
    stack: str
    count: int
    healthy_count: int
    probes: list[IntegrationHealthItem] = Field(default_factory=list)


IntegrationHealthStack = Literal["lab", "m6_p4", "all"]


def _collect_probes(stack: IntegrationHealthStack) -> tuple[str, list[IntegrationHealthItem]]:
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations()
    probes: list[IntegrationHealthItem] = []
    if stack in {"lab", "all"}:
        for item in health_check_harness_lab_stack():
            probes.append(
                IntegrationHealthItem(slug=item.slug, healthy=item.healthy, detail=item.detail)
            )
    if stack in {"m6_p4", "all"}:
        for item in health_check_harness_m6_p4_probes():
            probes.append(
                IntegrationHealthItem(slug=item.slug, healthy=item.healthy, detail=item.detail)
            )
    return stack, probes


def create_integration_health_debug_router() -> APIRouter:
    """Create read-only integration health endpoints for harness operators."""
    router = APIRouter(prefix="/debug/integrations", tags=["debug-integrations"])

    @router.get("/health", response_model=IntegrationHealthResponse)
    def integration_health(
        stack: IntegrationHealthStack = Query(
            default="all",
            description="Probe lab stable stack, M.6 P4 ROI slugs, or both",
        ),
    ) -> IntegrationHealthResponse:
        resolved_stack, probes = _collect_probes(stack)
        healthy_count = sum(1 for item in probes if item.healthy)
        return IntegrationHealthResponse(
            stack=resolved_stack,
            count=len(probes),
            healthy_count=healthy_count,
            probes=probes,
        )

    return router
