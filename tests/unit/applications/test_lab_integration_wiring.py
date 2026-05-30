# © Artur Czarnecki. All rights reserved.

"""Unit tests for lab IntegrationProfile wiring (Phase M.8)."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.log.adapter import LogNotificationAdapter
from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from lab_application.host.integration_wiring import (
    build_lab_integration_profile,
    wire_lab_integrations,
)
from lab_application.host.settings import LabApplicationSettings

pytestmark = pytest.mark.unit


def test_build_lab_integration_profile_defaults() -> None:
    profile = build_lab_integration_profile()
    assert profile.relational_store == IntegrationSlug.SQLITE
    assert profile.notification_channel == IntegrationSlug.LOG
    assert profile.interaction_surface == IntegrationSlug.LAB_JSON


def test_wire_lab_integrations_uses_profile_and_sqlite(tmp_path) -> None:
    settings = LabApplicationSettings()
    wiring = wire_lab_integrations(
        settings=settings,
        db_path=tmp_path / "trace.db",
        experiments_db_path=tmp_path / "experiments.db",
        runtime_events_db_path=tmp_path / "events.db",
        checkpoints_db_path=tmp_path / "checkpoints.db",
    )

    assert wiring.profile.relational_store == IntegrationSlug.SQLITE
    assert isinstance(wiring.notification_adapter, LogNotificationAdapter)
    assert wiring.checkpoint_store is wiring.sqlite_bundle.task_checkpoint_store
    assert wiring.runtime_event_store is wiring.sqlite_bundle.runtime_event_store


def test_wire_lab_integrations_in_memory_trace_when_no_db_path() -> None:
    wiring = wire_lab_integrations(settings=LabApplicationSettings())

    assert isinstance(wiring.trace_store, InMemoryRunTraceStore)
    assert wiring.trace_db_path is None
    assert wiring.runtime_event_store is None
