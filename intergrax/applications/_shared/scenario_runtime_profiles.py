# © Artur Czarnecki. All rights reserved.

"""Scenario runtime LAB and production-attached profile helpers (SCENARIO-PLATFORM-4)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.registry.agent_registry import AgentRegistry


class ScenarioRuntimeMode(str, Enum):
    """Canonical scenario runtime storage posture."""

    LAB = "lab"
    PRODUCTION_ATTACHED = "production_attached"


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeWorkspace:
    """LAB runtime storage root, derived SQLite paths, and root ownership."""

    root: Path
    runtime_events_db_path: Path
    trace_db_path: Path
    owns_root: bool


def create_scenario_lab_workspace(
    workspace_root: Path | None = None,
    *,
    prefix: str = "intergrax-scenario-lab-",
) -> ScenarioRuntimeWorkspace:
    """
    Create an isolated LAB workspace with canonical SQLite filenames.

    When ``workspace_root`` is omitted the platform creates a temp directory
    (``owns_root=True``). When a caller supplies ``workspace_root`` the platform
    does not own that directory (``owns_root=False``).
    """
    owns_root = workspace_root is None
    root = workspace_root if workspace_root is not None else Path(mkdtemp(prefix=prefix))
    root.mkdir(parents=True, exist_ok=True)
    return ScenarioRuntimeWorkspace(
        root=root,
        runtime_events_db_path=root / "runtime_events.db",
        trace_db_path=root / "trace.db",
        owns_root=owns_root,
    )


def cleanup_scenario_runtime_workspace(workspace: ScenarioRuntimeWorkspace) -> None:
    """Remove a platform-owned LAB workspace root when no longer needed."""
    if workspace.owns_root:
        shutil.rmtree(workspace.root, ignore_errors=True)


def build_scenario_lab_runtime(
    *,
    registry: AgentRegistry | None = None,
    tenant_id: str,
    profile_id: str = "scenario.lab",
    scenario_slug: str | None = None,
    workspace_root: Path | None = None,
    manifest: ApplicationManifest | None = None,
    document_store: Any | None = None,
    settings: Any = None,
) -> Any:
    """
    Build a zero-config LAB scenario runtime with local automatic storage.

    Authors do not pass runtime event / trace DB paths or persistence flags.
    LAB uses the same Nexus baseline as production-attached; only storage defaults differ.
    """
    from intergrax.applications._shared.scenario_runtime_baseline import (
        build_scenario_runtime_from_environment,
    )

    workspace = create_scenario_lab_workspace(workspace_root)
    resolved_profile_id = profile_id
    if scenario_slug is not None:
        resolved_profile_id = f"{scenario_slug}.lab"
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id=resolved_profile_id)
    resolved_document_store = (
        document_store if document_store is not None else InMemoryDocumentStore()
    )
    resolved_registry = registry if registry is not None else AgentRegistry()
    return build_scenario_runtime_from_environment(
        environment=environment,
        registry=resolved_registry,
        tenant_id=tenant_id,
        manifest=manifest,
        runtime_events_db_path=workspace.runtime_events_db_path,
        trace_db_path=workspace.trace_db_path,
        document_store=resolved_document_store,
        settings=settings,
        use_in_memory_trace=False,
        require_runtime_event_persistence=True,
        workspace=workspace,
        runtime_mode=ScenarioRuntimeMode.LAB,
    )


def build_scenario_production_runtime(
    *,
    environment: ApplicationEnvironmentProfile,
    manifest: ApplicationManifest,
    registry: AgentRegistry,
    tenant_id: str,
    runtime_events_db_path: Path,
    trace_db_path: Path | None = None,
    document_store: Any | None = None,
    settings: Any = None,
) -> Any:
    """
    Build a production-attached scenario runtime with explicit durable configuration.

    Missing manifest, tenant, storage, or required diagnostics prerequisites fail closed.
    """
    from intergrax.applications._shared.scenario_runtime_baseline import (
        ScenarioRuntimeBuildError,
        build_scenario_runtime_from_environment,
        validate_scenario_tenant_id,
    )

    if manifest is None:
        raise ScenarioRuntimeBuildError(
            "explicit ApplicationManifest is required for production-attached scenario runtime"
        )
    validate_scenario_tenant_id(tenant_id)
    if runtime_events_db_path is None:
        raise ScenarioRuntimeBuildError(
            "runtime_events_db_path is required for production-attached scenario runtime"
        )

    return build_scenario_runtime_from_environment(
        environment=environment,
        registry=registry,
        tenant_id=tenant_id,
        manifest=manifest,
        runtime_events_db_path=runtime_events_db_path,
        trace_db_path=trace_db_path,
        document_store=document_store,
        settings=settings,
        use_in_memory_trace=False,
        require_runtime_event_persistence=True,
        workspace=None,
        runtime_mode=ScenarioRuntimeMode.PRODUCTION_ATTACHED,
    )


__all__ = [
    "ScenarioRuntimeMode",
    "ScenarioRuntimeWorkspace",
    "build_scenario_lab_runtime",
    "build_scenario_production_runtime",
    "cleanup_scenario_runtime_workspace",
    "create_scenario_lab_workspace",
]
