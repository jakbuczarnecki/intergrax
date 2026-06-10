# © Artur Czarnecki. All rights reserved.

"""HTTP routes for GOV-PROD.1 unified product observability dashboard."""

from __future__ import annotations

from fastapi import APIRouter

from intergrax.runtime.observability.product_observability_dashboard import ProductObservabilityDashboard


def create_product_observability_dashboard_router(
    *,
    dashboard: ProductObservabilityDashboard,
    enabled: bool = True,
) -> APIRouter:
    router = APIRouter(prefix="/ops/dashboard", tags=["observability-dashboard"])

    @router.get("/health")
    def dashboard_health() -> dict[str, str]:
        if not enabled:
            return {"status": "disabled"}
        return {"status": "ok", "host_id": dashboard.host_id}

    @router.get("/unified")
    def unified_dashboard() -> dict[str, object]:
        if not enabled:
            return {"enabled": False}
        return {"enabled": True, "dashboard": dashboard.model_dump()}

    @router.get("/governance")
    def governance_dashboard() -> dict[str, object]:
        if not enabled:
            return {"enabled": False}
        return {"enabled": True, "governance": dashboard.governance.model_dump()}

    return router
