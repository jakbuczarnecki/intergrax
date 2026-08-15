# © Artur Czarnecki. All rights reserved.

"""MEM-4.3: Full memory stack on lab profile (task + session + LTM + org)."""

from __future__ import annotations

from pathlib import Path

import pytest
from lab_application.host.settings import LabApplicationSettings

from intergrax.agents.reference_harness import default_reference_harness
from intergrax.applications._shared.lab_environment_profile import (
    build_lab_environment_profile,
)
from intergrax.applications._shared.memory_wiring import (
    build_session_manager_from_environment,
    resolve_memory_platform_wiring,
)
from intergrax.applications._shared.runtime_config_bridge import (
    materialize_runtime_config,
)
from intergrax.applications._shared.task_memory_wiring import (
    wire_task_memory_from_profile,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    create_sqlite_integration,
)
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
    SQLiteOrganizationProfileStore,
)
from intergrax.runtime.task_memory.stores.sqlite_task_memory_store import (
    SQLiteTaskMemoryStore,
)
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def test_lab_profile_memory_flags_enabled() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)

    assert env.memory_profile.enable_user_memory is True
    assert env.memory_profile.enable_org_memory is True
    assert env.memory_profile.enable_long_term_memory is True
    assert env.memory_profile.enable_task_memory is True


def test_lab_profile_resolves_sqlite_memory_backends(tmp_path: Path) -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }

    wiring = resolve_memory_platform_wiring(env)

    assert wiring.sqlite_bundle is not None
    assert isinstance(wiring.session_storage, SQLiteSessionStorage)
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)
    assert isinstance(wiring.organization_profile_store, SQLiteOrganizationProfileStore)


def test_lab_profile_materializes_runtime_config_memory_toggles(tmp_path: Path) -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    request = RuntimeRequest(
        tenant_id="lab",
        agent_id="echo",
        user_id="tester",
        session_id="sess-lab",
        message="memory stack probe",
    )

    config = materialize_runtime_config(
        request,
        default_reference_harness(),
        env,
        llm_adapter=FakeLLMAdapter(fixed_text="ok"),
    )

    assert config.enable_user_profile_memory is True
    assert config.enable_org_profile_memory is True
    assert config.enable_user_longterm_memory is True
    assert config.enable_task_memory is True


def test_lab_profile_wires_task_session_and_profile_stores(tmp_path: Path) -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }

    bundle = create_sqlite_integration(data_dir=tmp_path)
    task_wiring = wire_task_memory_from_profile(env, db_path=bundle.paths.task_memory)
    session_manager = build_session_manager_from_environment(
        env,
        memory_wiring=resolve_memory_platform_wiring(env),
        tenant_id="lab",
    )

    assert task_wiring.store is not None
    assert isinstance(task_wiring.store, SQLiteTaskMemoryStore)
    assert session_manager._user_profile_manager is not None
    assert session_manager._organization_profile_manager is not None
    assert isinstance(session_manager._storage, SQLiteSessionStorage)
