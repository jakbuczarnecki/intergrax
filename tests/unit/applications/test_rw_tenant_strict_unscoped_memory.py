# © Artur Czarnecki. All rights reserved.
"""T-IGX-051-A Part H: STRICT forbids unscoped memory (EX-D2-06)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.applications._shared.memory_provider_admission import MemoryTenantScopeError
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.applications.contracts.environment_profile.presets import harness_memory_profile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.persistence.sqlite_composition import create_sqlite_runtime_persistence

pytestmark = pytest.mark.unit

_TENANT = "tenant-alpha"


def _strict_env(memory: MemoryProfile) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rw.ten.08")
    env.integration_profile = IntegrationProfile()
    env.execution_mode = ExecutionMode.STRICT
    env.memory_profile = memory
    return env


def test_ten_08a_strict_unscoped_memory_refused() -> None:
    env = _strict_env(harness_memory_profile())
    with pytest.raises(MemoryTenantScopeError):
        resolve_memory_platform_wiring(env)


def test_ten_08b_strict_with_tenant_composes(tmp_path: Path) -> None:
    create_sqlite_runtime_persistence(data_dir=tmp_path)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rw.ten.08")
    env.execution_mode = ExecutionMode.STRICT
    env.memory_profile = harness_memory_profile()
    wiring = resolve_memory_platform_wiring(env, tenant_id=_TENANT)
    assert wiring.session_storage is not None


def test_ten_08c_lab_host_unchanged() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rw.ten.08.lab")
    env.integration_profile = IntegrationProfile()
    env.memory_profile = harness_memory_profile()
    wiring = resolve_memory_platform_wiring(env)
    assert wiring.session_storage is not None


def test_ten_08d_memory_disabled_no_refusal() -> None:
    env = _strict_env(MemoryProfile())
    wiring = resolve_memory_platform_wiring(env)
    assert wiring.session_storage is not None


def test_ten_08e_product_default_flags_not_refused() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="rw.ten.08.product")
    env.integration_profile = IntegrationProfile()
    env.execution_mode = ExecutionMode.STRICT
    wiring = resolve_memory_platform_wiring(env)
    assert wiring.session_storage is not None


def test_ten_08f_session_flags_not_counted() -> None:
    env = _strict_env(MemoryProfile(enable_session_vector_index=True))
    wiring = resolve_memory_platform_wiring(env)
    assert wiring.session_storage is not None
