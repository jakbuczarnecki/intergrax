# © Artur Czarnecki. All rights reserved.

"""REL-1/2: Reliability runtime bridge and assembly validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.reliability_assembly_resolver import (
    ReliabilityAssemblyError,
    assert_reliability_assembly_valid,
    validate_reliability_wiring,
)
from intergrax.applications._shared.reliability_runtime_bridge import (
    resolve_reliability_wiring_options,
)
from intergrax.applications._shared.reliability_wiring import wire_application_reliability
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
    ReliabilityProfile,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_resolve_reliability_wiring_options_maps_profile_fields() -> None:
    profile = ReliabilityProfile(
        idempotency_enabled=False,
        circuit_breaker_failure_threshold=3,
        checkpoint_interval_steps=2,
        long_running_scheduler_enabled=True,
    )
    options = resolve_reliability_wiring_options(profile)
    assert options.idempotency_enabled is False
    assert options.circuit_breaker_failure_threshold == 3
    assert options.checkpoint_interval_steps == 2
    assert options.long_running_scheduler_enabled is True


def test_wire_application_reliability_uses_sqlite_when_path_provided(tmp_path) -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.sqlite")
    wiring = wire_application_reliability(env, idempotency_db_path=tmp_path / "idempotency.db")
    assert isinstance(wiring.idempotency_store, SQLiteIdempotencyStore)
    assert wiring.circuit_breaker_config.failure_threshold == env.reliability_profile.circuit_breaker_failure_threshold


def test_wire_application_reliability_uses_in_memory_when_idempotency_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.mem")
    env.reliability_profile = ReliabilityProfile(idempotency_enabled=False)
    wiring = wire_application_reliability(env)
    assert wiring.idempotency_store is None


def test_assert_reliability_assembly_valid_lab_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.valid")
    wiring = wire_application_reliability(env)
    assert_reliability_assembly_valid(wiring, env)


def test_validate_reliability_wiring_requires_orchestration_long_running() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.orch")
    env.reliability_profile = ReliabilityProfile(long_running_scheduler_enabled=True)
    env.orchestration_profile = OrchestrationProfile(long_running_enabled=False)
    wiring = wire_application_reliability(env)
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any("orchestration_profile.long_running_enabled" in error for error in result.errors)


def test_validate_reliability_wiring_rejects_store_when_idempotency_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.reject")
    env.reliability_profile = ReliabilityProfile(idempotency_enabled=False)
    wiring = wire_application_reliability(env)
    wiring = type(wiring)(
        options=wiring.options,
        idempotency_store=InMemoryIdempotencyStore(),
        circuit_breaker_config=wiring.circuit_breaker_config,
    )
    with pytest.raises(ReliabilityAssemblyError):
        assert_reliability_assembly_valid(wiring, env)


def test_materialize_runtime_config_applies_idempotency_store() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.runtime")
    request = RuntimeRequest(
        message="hello",
        tenant_id="t1",
        agent_id="echo",
        user_id="user-1",
        session_id="session-1",
    )
    manifest = build_lab_manifest(LabApplicationSettings.from_env())
    build_ctx = ApplicationBuildContext.for_manifest(manifest, environment=env)
    config = materialize_runtime_config(request, build_ctx, env)
    assert config.idempotency_store is not None
    assert isinstance(config.idempotency_store, InMemoryIdempotencyStore)


def test_build_harness_host_runtime_wires_reliability_artifacts() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    assert runtime.reliability.idempotency_store is not None
