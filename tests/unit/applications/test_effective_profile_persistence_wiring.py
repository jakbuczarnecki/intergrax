# © Artur Czarnecki. All rights reserved.

"""Focused durability wiring proofs for effective profile persistence (NPSC-3B-R3V-R1E)."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.profile_resolution.activation_store import (
    InMemoryActiveEffectiveProfileRevisionStore,
)
from intergrax.applications._shared.profile_resolution.execution_pinning import (
    InMemoryEffectiveProfileExecutionPinningStore,
)
from intergrax.applications._shared.profile_resolution.store import (
    InMemoryEffectiveProfileRevisionStore,
)
from intergrax.applications._shared.profile_resolution.wiring import (
    resolve_effective_profile_persistence_wiring,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
    resolve_harness_host_profile_persistence_kwargs_from_composition,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileRevisionError,
)
from intergrax.applications._shared.profile_resolution.persistence import (
    KvEffectiveProfileExecutionPinningStore,
    KvEffectiveProfileRevisionStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _echo_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="profile_persistence",
        name="Profile Persistence Host",
        route_prefix="/v1/profile_persistence",
        env_prefix="PROFILE_PERSISTENCE_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def _application(*, execution_mode: ExecutionMode = ExecutionMode.STRICT) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="profile_persistence")
    return profile.model_copy(
        update={"meta": profile.meta.model_copy(update={"execution_mode": execution_mode})},
    )


def test_production_without_durable_backing_fails_closed() -> None:
    with pytest.raises(
        EffectiveProfileRevisionError,
        match="durable effective profile revision store",
    ):
        resolve_effective_profile_persistence_wiring(production_mode=True)


def test_production_with_platform_persistence_succeeds() -> None:
    platform = build_reference_production_platform_persistence()
    wiring = resolve_effective_profile_persistence_wiring(
        production_mode=True,
        kv_store=platform.kv_store,
    )
    assert isinstance(wiring.revision_store, KvEffectiveProfileRevisionStore)
    assert isinstance(wiring.pinning_store, KvEffectiveProfileExecutionPinningStore)
    assert wiring.revision_store.is_durable is True
    assert wiring.pinning_store.is_durable is True


def test_non_production_allows_in_memory_without_backing() -> None:
    wiring = resolve_effective_profile_persistence_wiring(production_mode=False)
    assert isinstance(wiring.revision_store, InMemoryEffectiveProfileRevisionStore)
    assert isinstance(wiring.pinning_store, InMemoryEffectiveProfileExecutionPinningStore)
    assert wiring.revision_store.is_durable is False
    assert wiring.pinning_store.is_durable is False


def test_shared_composition_reuses_platform_persistence_authority() -> None:
    composition = create_reference_production_process_composition()
    first = resolve_harness_host_profile_persistence_kwargs_from_composition(
        production_mode=True,
        composition=composition,
    )
    second = resolve_harness_host_profile_persistence_kwargs_from_composition(
        production_mode=True,
        composition=composition,
    )
    assert first["key_value_cache"] is second["key_value_cache"]


def test_explicit_non_durable_revision_store_fails_closed_in_production() -> None:
    with pytest.raises(
        EffectiveProfileRevisionError,
        match="durable effective profile revision store",
    ):
        resolve_effective_profile_persistence_wiring(
            production_mode=True,
            revision_store=InMemoryEffectiveProfileRevisionStore(),
            pinning_store=InMemoryEffectiveProfileExecutionPinningStore(),
            active_store=InMemoryActiveEffectiveProfileRevisionStore(),
        )
