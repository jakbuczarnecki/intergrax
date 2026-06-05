# © Artur Czarnecki. All rights reserved.

"""OBS-1/2: Observability runtime bridge and assembly validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.observability_assembly_resolver import (
    ObservabilityAssemblyError,
    assert_observability_assembly_valid,
    validate_observability_wiring,
)
from intergrax.applications._shared.observability_runtime_bridge import (
    resolve_observability_wiring_options,
)
from intergrax.applications._shared.observability_wiring import wire_application_observability
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ObservabilityProfile,
)
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_observability_wiring_options_maps_trace_sqlite() -> None:
    enabled = resolve_observability_wiring_options(
        ObservabilityProfile(trace_sqlite_enabled=True),
    )
    disabled = resolve_observability_wiring_options(
        ObservabilityProfile(trace_sqlite_enabled=False),
    )
    assert enabled.use_in_memory_trace is False
    assert enabled.enable_runtime_events is True
    assert disabled.use_in_memory_trace is True
    assert disabled.enable_runtime_events is False


def test_wire_application_observability_uses_sqlite_when_enabled(tmp_path) -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="obs.sqlite")
    wiring = wire_application_observability(
        env,
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
    )
    assert isinstance(wiring.stores.trace_store, SQLiteRunTraceStore)
    assert wiring.stores.runtime_event_store is not None


def test_wire_application_observability_uses_in_memory_when_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="obs.mem")
    env.observability_profile = ObservabilityProfile(trace_sqlite_enabled=False)
    wiring = wire_application_observability(env)
    assert isinstance(wiring.stores.trace_store, InMemoryRunTraceStore)
    assert wiring.stores.runtime_event_store is None


def test_assert_observability_assembly_valid_lab_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="obs.valid")
    wiring = wire_application_observability(env)
    assert_observability_assembly_valid(wiring, env)


def test_validate_observability_wiring_requires_backend_when_otel_enabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="obs.otel")
    env.observability_profile = ObservabilityProfile(otel_enabled=True)
    wiring = wire_application_observability(env)
    result = validate_observability_wiring(wiring, env)
    if env.integration_profile.observability_backend is None:
        assert not result.valid
        assert any("observability_backend" in error for error in result.errors)
    else:
        assert result.valid


def test_validate_observability_wiring_rejects_in_memory_when_sqlite_enabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="obs.reject")
    stores = wire_nexus_observability(use_in_memory_trace=True, enable_runtime_events=False)
    wiring = wire_application_observability(env)
    wiring = type(wiring)(options=wiring.options, stores=stores)
    with pytest.raises(ObservabilityAssemblyError):
        assert_observability_assembly_valid(wiring, env)


def test_build_harness_host_runtime_wires_observability_stores() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    assert runtime.observability.trace_store is not None
