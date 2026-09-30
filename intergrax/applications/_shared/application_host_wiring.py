# © Artur Czarnecki. All rights reserved.

"""Mount Tier-3 :class:`ApplicationHost` on Nexus middleware (APP-CON-1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.middleware.base import RuntimeMiddleware

if TYPE_CHECKING:
    from intergrax.harness.application_host import ApplicationHost
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications.contracts.manifest import ApplicationManifest
    from intergrax.contracts.run_budget import RunBudget
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)


def _attach_middleware(
    host: HostOrchestrationApplicationWiringTarget,
    middleware: RuntimeMiddleware,
) -> None:
    pipeline = host.middleware
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
    host: HostOrchestrationApplicationWiringTarget,
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
        host,
        ApplicationEnvironmentStateMiddleware(
            manifest=manifest,
            environment=environment,
            run_budget=run_budget,
        ),
    )


def apply_application_host_wiring(
    orchestration_host: HostOrchestrationApplicationWiringTarget,
    application_host: ApplicationHost | None,
) -> None:
    """Attach ``ApplicationHost`` middleware when a host implementation is provided."""
    if application_host is None:
        return
    from intergrax.harness.hooks import ApplicationHostMiddleware

    _attach_middleware(orchestration_host, ApplicationHostMiddleware(application_host))


def apply_hook_runtime_guard_wiring(
    host: HostOrchestrationApplicationWiringTarget,
    environment: ApplicationEnvironmentProfile,
) -> None:
    """Configure middleware hook timeout and audit bus (APP-CON-5 · §32.6.5)."""
    pipeline = host.middleware
    if not isinstance(pipeline, MiddlewarePipeline):
        return
    pipeline.configure_hook_runtime(
        hook_timeout_seconds=environment.reliability_profile.middleware_hook_timeout_seconds,
        event_bus=host.event_bus,
    )
