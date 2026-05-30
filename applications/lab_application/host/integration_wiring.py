# © Artur Czarnecki. All rights reserved.

"""Lab Tier-3 integration composition via ``IntegrationProfile`` (Phase M.8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.sqlite.bundle import (
    SQLiteIntegrationBundle,
    create_sqlite_integration,
)
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.factory import (
    InteractionSurface,
    create_interaction_adapter,
    resolve_interaction_settings,
)
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from lab_application.host.settings import LabApplicationSettings


@dataclass(frozen=True)
class LabIntegrationWiring:
    """Resolved integrations for the lab application factory."""

    profile: IntegrationProfile
    sqlite_bundle: SQLiteIntegrationBundle
    trace_store: RunTraceWriter
    runtime_event_store: RuntimeEventPersistence | None
    checkpoint_store: TaskCheckpointPersistence
    notification_adapter: NotificationAdapter
    interaction_adapter: InteractionAdapter
    trace_db_path: Path | None
    runtime_events_db_path: Path | None
    experiments_db_path: Path | None
    checkpoints_db_path: Path | None


def _sqlite_config_overrides(
    *,
    db_path: Path | None,
    experiments_db_path: Path | None,
    runtime_events_db_path: Path | None,
    checkpoints_db_path: Path | None,
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    if db_path is not None:
        overrides["trace_db"] = db_path
    if experiments_db_path is not None:
        overrides["experiments_db"] = experiments_db_path
    if runtime_events_db_path is not None:
        overrides["runtime_events_db"] = runtime_events_db_path
    if checkpoints_db_path is not None:
        overrides["task_checkpoints_db"] = checkpoints_db_path
    return overrides


def build_lab_integration_profile(
    *,
    sqlite_overrides: dict[str, Path] | None = None,
) -> IntegrationProfile:
    profile = IntegrationProfile.lab()
    if not sqlite_overrides:
        return profile
    return profile.model_copy(
        update={
            "options": {
                IntegrationSlug.SQLITE: dict(sqlite_overrides),
            }
        }
    )


def create_lab_interaction_adapter(settings: LabApplicationSettings) -> InteractionAdapter:
    surface = settings.interaction_surface.strip().lower()
    if surface in {InteractionSurface.LAB.value, InteractionSurface.LAB_JSON.value}:
        profile = IntegrationProfile.lab()
        return profile.resolve(IntegrationCategory.INTERACTION_SURFACE)
    return create_interaction_adapter(
        resolve_interaction_settings(surface=surface or InteractionSurface.AUTO.value)
    )


def wire_lab_integrations(
    *,
    settings: LabApplicationSettings,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
) -> LabIntegrationWiring:
    """
    Single composition root for lab persistence, notifications, and interaction surface.

    Uses ``IntegrationProfile.lab()`` (sqlite + log + lab_json) with optional SQLite path overrides.
    """
    register_default_integrations()

    sqlite_overrides = _sqlite_config_overrides(
        db_path=db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
    )
    profile = build_lab_integration_profile(sqlite_overrides=sqlite_overrides or None)
    sqlite_bundle = create_sqlite_integration(**sqlite_overrides)

    if db_path is None:
        trace_store: RunTraceWriter = InMemoryRunTraceStore()
        trace_db_path = None
    else:
        trace_store = sqlite_bundle.trace_store  # type: ignore[assignment]
        trace_db_path = db_path

    runtime_event_store = (
        sqlite_bundle.runtime_event_store
        if runtime_events_db_path is not None
        else None
    )

    notification_adapter = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
    interaction_adapter = create_lab_interaction_adapter(settings)

    return LabIntegrationWiring(
        profile=profile,
        sqlite_bundle=sqlite_bundle,
        trace_store=trace_store,
        runtime_event_store=runtime_event_store,  # type: ignore[arg-type]
        checkpoint_store=sqlite_bundle.task_checkpoint_store,  # type: ignore[arg-type]
        notification_adapter=notification_adapter,  # type: ignore[arg-type]
        interaction_adapter=interaction_adapter,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        experiments_db_path=experiments_db_path or sqlite_bundle.paths.experiments,
        checkpoints_db_path=checkpoints_db_path or sqlite_bundle.paths.task_checkpoints,
    )
