# © Artur Czarnecki. All rights reserved.

"""Mount Tier-3 :class:`ApplicationHost` on Nexus middleware (APP-CON-1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.middleware.base import RuntimeMiddleware

if TYPE_CHECKING:
    from intergrax.harness.application_host import ApplicationHost
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications.contracts.manifest import ApplicationManifest
    from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def _attach_middleware(nexus: NexusLoop, middleware: RuntimeMiddleware) -> None:
    pipeline = nexus._middleware  # noqa: SLF001 — Tier-3 composition hook
    if not isinstance(pipeline, MiddlewarePipeline):
        return
    existing = list(pipeline._middleware)  # noqa: SLF001
    if any(mw.name == middleware.name for mw in existing):
        return
    pipeline._middleware = sorted(  # noqa: SLF001
        [*existing, middleware],
        key=lambda item: item.priority,
    )


def apply_application_environment_state_wiring(
    nexus: NexusLoop,
    *,
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    run_budget: RunBudget | None = None,
) -> None:
    """Attach lifecycle sync middleware for ``ApplicationEnvironmentState`` (APP-CON-3)."""
    from intergrax.applications._shared.application_environment_state_middleware import (
        ApplicationEnvironmentStateMiddleware,
    )

    _attach_middleware(
        nexus,
        ApplicationEnvironmentStateMiddleware(
            manifest=manifest,
            environment=environment,
            run_budget=run_budget,
        ),
    )


def apply_application_host_wiring(
    nexus: NexusLoop,
    host: ApplicationHost | None,
) -> None:
    """Attach ``ApplicationHost`` middleware when a host implementation is provided."""
    if host is None:
        return
    from intergrax.harness.hooks import ApplicationHostMiddleware

    _attach_middleware(nexus, ApplicationHostMiddleware(host))


def apply_hook_runtime_guard_wiring(
    nexus: NexusLoop,
    environment: ApplicationEnvironmentProfile,
) -> None:
    """Configure middleware hook timeout and audit bus (APP-CON-5 · §32.6.5)."""
    pipeline = nexus._middleware  # noqa: SLF001 — Tier-3 composition hook
    if not isinstance(pipeline, MiddlewarePipeline):
        return
    pipeline.configure_hook_runtime(
        hook_timeout_seconds=environment.reliability_profile.middleware_hook_timeout_seconds,
        event_bus=nexus.event_bus,
    )
