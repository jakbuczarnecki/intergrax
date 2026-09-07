# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background worker process entrypoint (LKW.4E)."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from intergrax.integrations.contracts.document_store import DocumentStore

from intergrax.applications._shared.diagnostic_read_wiring import (
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    build_diagnostic_orchestrator,
)
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedDiagnosticTenantBinding,
    build_hosted_application_diagnostic_event_publisher,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_runtime_materialization import (
    prepare_reference_runtime_materialization,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.hosting import (
    HostedProcessBootstrapContext,
    HostedProcessBootstrapPhase,
    run_guarded_hosted_process_bootstrap,
)
from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from local_workspace_application.host.background_worker_constructor import (
    BackgroundWorkerConstructor,
    BlockingBackgroundWorker,
)
from local_workspace_application.host.background_worker_factory import (
    LocalWorkspaceBackgroundWorkerWiring,
    build_local_workspace_background_worker_wiring,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.message_bus_wiring import local_workspace_message_bus_enabled
from local_workspace_application.host.observability_wiring import (
    resolve_local_workspace_observability_exporter,
)
from local_workspace_application.host.reference_lifecycle_input import (
    build_local_workspace_reference_lifecycle_input,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)

logger = logging.getLogger(__name__)

_BACKGROUND_WORKER_PROCESS_ROLE = "background_worker"


@dataclass(frozen=True, slots=True)
class LocalWorkspaceWorkerBootstrapDiagnostics:
    event_publisher: HostedApplicationEventPublisher
    diagnostic_tenant: HostedDiagnosticTenantBinding


def _resolve_settings(
    settings: LocalWorkspaceBackendSettings | None,
) -> LocalWorkspaceBackendSettings:
    if settings is not None:
        return settings
    candidate_settings = LocalWorkspaceBackendSettings.from_env()
    if not isinstance(candidate_settings, LocalWorkspaceBackendSettings):
        raise TypeError("local workspace settings factory returned an invalid type")
    return candidate_settings


def activate_local_workspace_reference_production_authority(
    settings: LocalWorkspaceBackendSettings | None = None,
    *,
    environment_profile: ApplicationEnvironmentProfile | None = None,
) -> tuple[ProductionProcessComposition, MaterializedRegistryProjection]:
    """Deploy/activate reference production lifecycle and resolve registry projection."""
    resolved_settings = _resolve_settings(settings)
    env = environment_profile or build_local_workspace_environment_profile(resolved_settings)
    composition = create_reference_production_process_composition()
    projection_input, activation_request = build_local_workspace_reference_lifecycle_input(
        resolved_settings,
    )
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    prepare_reference_runtime_materialization(
        composition.agent_platform_runtime.stores,
        projection_input,
        artifact_locator=activation_request.artifact_locator,
    )
    launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
        admission_mutation_id=reference_admission_mutation_id(
            projection_input.runtime_revision.runtime_revision_id
        ),
    )
    registry_projection = bootstrap_production_registry_projection(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    return composition, registry_projection


def _build_worker_diagnostic_runtime(
    *,
    registry_projection: MaterializedRegistryProjection,
    settings: LocalWorkspaceBackendSettings,
    document_store: DocumentStore,
) -> HarnessHostRuntime:
    manifest = LOCAL_WORKSPACE_APPLICATION_MANIFEST
    return build_harness_host_runtime(
        manifest,
        manifest.resolved_environment(),
        settings=settings,
        idempotency_db_path=Path(settings.idempotency_db_path),
        document_store=document_store,
        registry_projection=registry_projection,
    )


def build_local_workspace_worker_bootstrap_diagnostics(
    *,
    registry_projection: MaterializedRegistryProjection,
    settings: LocalWorkspaceBackendSettings,
    environment_profile: ApplicationEnvironmentProfile,
    document_store: DocumentStore,
) -> LocalWorkspaceWorkerBootstrapDiagnostics:
    tenant_binding = HostedDiagnosticTenantBinding(
        tenant_id=environment_profile.profile_id,
    )
    runtime = _build_worker_diagnostic_runtime(
        registry_projection=registry_projection,
        settings=settings,
        document_store=document_store,
    )
    dependencies = resolve_host_diagnostic_read_dependencies(runtime)
    orchestrator = build_diagnostic_orchestrator(dependencies)
    observability_exporter = resolve_local_workspace_observability_exporter(settings)
    event_publisher = build_hosted_application_diagnostic_event_publisher(
        tenant_binding=tenant_binding,
        orchestrator=orchestrator,
        observability_exporter=observability_exporter,
    )
    return LocalWorkspaceWorkerBootstrapDiagnostics(
        event_publisher=event_publisher,
        diagnostic_tenant=tenant_binding,
    )


async def _run_guarded_worker_bootstrap(
    *,
    bootstrap_context: HostedProcessBootstrapContext,
    event_publisher: HostedApplicationEventPublisher,
    settings: LocalWorkspaceBackendSettings,
    registry_projection: MaterializedRegistryProjection,
    document_store: DocumentStore,
    worker_constructor: BackgroundWorkerConstructor | None = None,
) -> LocalWorkspaceBackgroundWorkerWiring:
    wiring = await run_guarded_hosted_process_bootstrap(
        context=bootstrap_context,
        phase=HostedProcessBootstrapPhase.WORKER_CONSTRUCTION,
        event_publisher=event_publisher,
        bootstrap=lambda: build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=registry_projection,
            settings=settings,
            document_store=document_store,
            worker_constructor=worker_constructor,
        ),
    )
    logger.info("Starting LKW Kafka background worker for lkw.background_ingest.v1")
    worker = wiring.worker
    if not isinstance(worker, BlockingBackgroundWorker):
        raise TypeError("background worker wiring returned an invalid worker type")
    await run_guarded_hosted_process_bootstrap(
        context=bootstrap_context,
        phase=HostedProcessBootstrapPhase.STARTUP,
        event_publisher=event_publisher,
        bootstrap=worker.start,
    )
    return wiring


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not local_workspace_message_bus_enabled():
        logger.error("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS must be true for the background worker")
        return 1

    settings = _resolve_settings(None)
    environment_profile = build_local_workspace_environment_profile(settings)
    _, registry_projection = activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    document_store = resolve_lkw_runtime_document_store(settings)
    bootstrap_diagnostics = build_local_workspace_worker_bootstrap_diagnostics(
        registry_projection=registry_projection,
        settings=settings,
        environment_profile=environment_profile,
        document_store=document_store,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_BACKGROUND_WORKER_PROCESS_ROLE,
    )
    asyncio.run(
        _run_guarded_worker_bootstrap(
            bootstrap_context=bootstrap_context,
            event_publisher=bootstrap_diagnostics.event_publisher,
            settings=settings,
            registry_projection=registry_projection,
            document_store=document_store,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
