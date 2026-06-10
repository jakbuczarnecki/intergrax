# © Artur Czarnecki. All rights reserved.

"""Trace Explorer wiring for product hosts (AUDIT-IDEAL-27.1)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter

from intergrax.applications._shared.trace_explorer_routes import create_trace_explorer_router
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class TraceExplorerWiring:
    enabled: bool
    router: APIRouter | None


def resolve_trace_explorer_wiring(env: ApplicationEnvironmentProfile) -> TraceExplorerWiring:
    """Mount trace explorer routes on product hosts when enabled."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return TraceExplorerWiring(enabled=False, router=None)
    if not env.features.trace_explorer_enabled:
        return TraceExplorerWiring(enabled=False, router=None)
    if not env.observability_profile.trace_sqlite_enabled:
        return TraceExplorerWiring(enabled=False, router=None)
    return TraceExplorerWiring(
        enabled=True,
        router=create_trace_explorer_router(enabled=True),
    )
