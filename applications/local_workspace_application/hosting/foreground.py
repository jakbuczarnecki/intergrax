# © Artur Czarnecki. All rights reserved.

"""LKW foreground hosted application facade (APP-HOST-8C)."""

from __future__ import annotations

from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedDiagnosticTenantBinding,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.hosting import (
    HostedApplicationSupervisorResult,
    run_hosted_application,
)
from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting.profile import (
    build_local_workspace_hosted_profile,
)


_STRICT_HOSTED_MESSAGE = (
    "LKW hosted foreground requires an activated ProductionProcessComposition. "
    "Pass process_composition= with lifecycle deploy/activate completed."
)


def run_local_workspace_hosted_application(
    *,
    process_composition: ProductionProcessComposition | None = None,
    settings: LocalWorkspaceBackendSettings | None = None,
    diagnostic_orchestrator: DiagnosticOrchestrator | None = None,
    diagnostic_tenant_binding: HostedDiagnosticTenantBinding | None = None,
) -> HostedApplicationSupervisorResult:
    """Build one LKW hosted profile and run it via the platform foreground runner."""
    if process_composition is None:
        raise HarnessHostRegistryAuthorityError(_STRICT_HOSTED_MESSAGE)
    profile = build_local_workspace_hosted_profile(
        process_composition=process_composition,
        settings=settings,
    )
    event_publisher_factory = None
    if diagnostic_orchestrator is not None and diagnostic_tenant_binding is not None:
        from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
            build_hosted_application_diagnostic_event_publisher,
        )

        tenant_binding = diagnostic_tenant_binding
        orchestrator = diagnostic_orchestrator

        def event_publisher_factory() -> HostedApplicationEventPublisher:
            return build_hosted_application_diagnostic_event_publisher(
                tenant_binding=tenant_binding,
                orchestrator=orchestrator,
            )

    elif diagnostic_orchestrator is not None or diagnostic_tenant_binding is not None:
        raise ValueError(
            "diagnostic_orchestrator and diagnostic_tenant_binding must be provided together",
        )
    return run_hosted_application(
        profile,
        event_publisher_factory=event_publisher_factory,
    )
