# © Artur Czarnecki. All rights reserved.

"""Integration composition for poc_template_application (lab profile)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
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
from poc_template_application.host.settings import PocTemplateApplicationSettings


@dataclass(frozen=True)
class PocTemplateIntegrationWiring:
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


def create_poc_template_interaction_adapter(
    settings: PocTemplateApplicationSettings,
) -> InteractionAdapter:
    surface = settings.interaction_surface.strip().lower()
    if surface in {InteractionSurface.LAB.value, InteractionSurface.LAB_JSON.value}:
        profile = IntegrationProfile.lab()
        return profile.resolve(IntegrationCategory.INTERACTION_SURFACE)
    return create_interaction_adapter(
        resolve_interaction_settings(surface=surface or InteractionSurface.AUTO.value)
    )


def wire_poc_template_integrations(
    *,
    settings: PocTemplateApplicationSettings,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
) -> PocTemplateIntegrationWiring:
    register_default_integrations()
    sqlite_overrides = _sqlite_config_overrides(
        db_path=db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
    )
    profile = IntegrationProfile.lab()
    if sqlite_overrides:
        profile = profile.model_copy(
            update={"options": {IntegrationSlug.SQLITE: dict(sqlite_overrides)}}
        )
    sqlite_bundle = create_sqlite_integration(**sqlite_overrides)
    if db_path is None:
        trace_store: RunTraceWriter = InMemoryRunTraceStore()
        trace_db_path = None
    else:
        trace_store = sqlite_bundle.trace_store  # type: ignore[assignment]
        trace_db_path = db_path
    runtime_event_store = (
        sqlite_bundle.runtime_event_store if runtime_events_db_path is not None else None
    )
    notification_adapter = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
    interaction_adapter = create_poc_template_interaction_adapter(settings)
    return PocTemplateIntegrationWiring(
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
