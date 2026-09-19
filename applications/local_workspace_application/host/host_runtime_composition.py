# © Artur Czarnecki. All rights reserved.

"""Canonical LKW harness host runtime + tenant authority composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.agent_certification_wiring import (
    apply_roster_agent_governance,
)
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.harness_registry_authority import (
    HarnessHostRegistryAuthorityError,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.contracts.document_store import DocumentStore
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.orchestration_decision_requirement_policy import (
    resolve_local_workspace_harness_orchestration_decision_requirement_policy,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)


class LocalWorkspaceHostRuntimeAuthorityError(ValueError):
    """LKW host tenant/runtime authority violation."""


@dataclass(frozen=True, slots=True)
class LocalWorkspaceHostTenantBinding:
    """Typed tenant authority for harness host runtime wiring."""

    tenant_id: str


@dataclass(frozen=True, slots=True)
class LocalWorkspaceHarnessHostRuntimeComposition:
    """Narrow harness runtime bundle produced by the canonical host composition root."""

    tenant_binding: LocalWorkspaceHostTenantBinding
    environment: ApplicationEnvironmentProfile
    document_store: DocumentStore
    runtime: HarnessHostRuntime


def build_local_workspace_host_environment(
    settings: LocalWorkspaceBackendSettings,
) -> ApplicationEnvironmentProfile:
    """Single settings-aware host environment profile (roster governance applied)."""
    manifest = LOCAL_WORKSPACE_APPLICATION_MANIFEST
    return apply_roster_agent_governance(
        build_local_workspace_environment_profile(settings),
        agents=manifest.agents,
        app_id=manifest.app_id,
    )


def _distinct_api_key_tenant_ids(settings: LocalWorkspaceBackendSettings) -> list[str]:
    tenant_ids: list[str] = []
    seen: set[str] = set()
    for identity in settings.api_keys_map.values():
        tenant_id = identity.tenant_id.strip()
        if tenant_id and tenant_id not in seen:
            seen.add(tenant_id)
            tenant_ids.append(tenant_id)
    return tenant_ids


def resolve_local_workspace_host_tenant_binding(
    settings: LocalWorkspaceBackendSettings,
) -> LocalWorkspaceHostTenantBinding:
    """Resolve host-level tenant authority from explicit host config or API key map."""
    configured_tenant = settings.host_tenant_id.strip()
    if configured_tenant:
        return LocalWorkspaceHostTenantBinding(tenant_id=configured_tenant)

    tenant_ids = _distinct_api_key_tenant_ids(settings)
    if len(tenant_ids) == 1:
        return LocalWorkspaceHostTenantBinding(tenant_id=tenant_ids[0])
    if len(tenant_ids) > 1:
        raise LocalWorkspaceHostRuntimeAuthorityError(
            "local_workspace_host_tenant_authority_ambiguous"
        )
    raise LocalWorkspaceHostRuntimeAuthorityError(
        "local_workspace_host_tenant_authority_missing"
    )


def build_local_workspace_harness_host_runtime(
    *,
    settings: LocalWorkspaceBackendSettings,
    registry_projection: MaterializedRegistryProjection,
    manifest: ApplicationManifest | None = None,
    environment: ApplicationEnvironmentProfile | None = None,
    tenant_binding: LocalWorkspaceHostTenantBinding | None = None,
    document_store: DocumentStore | None = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    idempotency_db_path: Path | None = None,
) -> LocalWorkspaceHarnessHostRuntimeComposition:
    """Compose harness host runtime with one tenant authority and configured providers."""
    if registry_projection is None:
        raise HarnessHostRegistryAuthorityError(
            "MaterializedRegistryProjection is required for LKW harness host runtime"
        )
    resolved_manifest = manifest or LOCAL_WORKSPACE_APPLICATION_MANIFEST
    resolved_environment = environment or build_local_workspace_host_environment(settings)
    resolved_document_store = (
        document_store
        if document_store is not None
        else resolve_lkw_runtime_document_store(settings)
    )
    resolved_tenant = tenant_binding or resolve_local_workspace_host_tenant_binding(
        settings,
    )
    resolved_idempotency = (
        idempotency_db_path
        if idempotency_db_path is not None
        else Path(settings.idempotency_db_path)
    )
    runtime = build_harness_host_runtime(
        resolved_manifest,
        resolved_environment,
        settings=settings,
        tenant_id=resolved_tenant.tenant_id,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        idempotency_db_path=resolved_idempotency,
        document_store=resolved_document_store,
        registry_projection=registry_projection,
        orchestration_decision_requirement_policy=(
            resolve_local_workspace_harness_orchestration_decision_requirement_policy(settings)
        ),
    )
    return LocalWorkspaceHarnessHostRuntimeComposition(
        tenant_binding=resolved_tenant,
        environment=resolved_environment,
        document_store=resolved_document_store,
        runtime=runtime,
    )


__all__ = [
    "LocalWorkspaceHarnessHostRuntimeComposition",
    "LocalWorkspaceHostRuntimeAuthorityError",
    "LocalWorkspaceHostTenantBinding",
    "build_local_workspace_harness_host_runtime",
    "build_local_workspace_host_environment",
    "resolve_local_workspace_host_tenant_binding",
]
