# © Artur Czarnecki. All rights reserved.

"""Sanctioned Governed Contractor strict production ASGI entry (PROD-Q-R1-R7-D4)."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.main import create_governed_contractor_process_app
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.applications._shared.strict_production_asgi_lifespan import (
    attach_strict_production_serving_lifespan,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingSession,
)


def create_governed_contractor_strict_production_asgi_app(
    session: StrictProductionServingSession,
    *,
    settings: GovernedContractorBackendSettings | None = None,
    host_runtime: GovernedContractorHostRuntimeComposition | None = None,
) -> FastAPI:
    """
    Build a traffic-ready GCA ASGI app from an opened strict production serving session.

    The session must already satisfy reconciliation, handoff binding, and admission wiring.
    """
    if not session.readiness.ready_for_traffic:
        raise ValueError(
            "strict production serving session is not ready for traffic; "
            "open via open_strict_production_serving_session first",
        )
    app = create_governed_contractor_process_app(
        process_composition=session.composition,
        production_governance=session.lifecycle.governance,
        settings=settings,
        host_runtime=host_runtime,
    )
    return attach_strict_production_serving_lifespan(app, session)


def build_governed_contractor_strict_production_uvicorn_factory(
    session: StrictProductionServingSession,
    *,
    settings: GovernedContractorBackendSettings | None = None,
    host_runtime: GovernedContractorHostRuntimeComposition | None = None,
) -> Callable[[], FastAPI]:
    """Uvicorn ``--factory`` entry that closes over one opened strict production session."""

    def _factory() -> FastAPI:
        return create_governed_contractor_strict_production_asgi_app(
            session,
            settings=settings,
            host_runtime=host_runtime,
        )

    return _factory


__all__ = [
    "build_governed_contractor_strict_production_uvicorn_factory",
    "create_governed_contractor_strict_production_asgi_app",
]
