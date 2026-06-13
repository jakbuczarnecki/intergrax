# © Artur Czarnecki. All rights reserved.

"""MEM closeout: MemoryProfile drives RuntimeConfig on harness reference hosts."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agents.reference_harness import LabHarnessContext, default_reference_harness
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from legal_application.host.wiring import build_legal_environment_profile
from legal_application.host.settings import LegalBackendSettings
from lab_application.host.settings import LabApplicationSettings
from poc_template_application.manifest import build_poc_template_manifest
from research_application.host.wiring import build_research_environment_profile
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-ref",
        agent_id="echo",
        user_id="user-ref",
        session_id="session-ref",
        message="reference host memory probe",
    )


@pytest.mark.parametrize(
    "env",
    [
        build_lab_environment_profile(LabApplicationSettings.from_env()),
        build_legal_environment_profile(LegalBackendSettings.from_env()),
        build_research_environment_profile(),
        build_poc_template_manifest().environment,
    ],
    ids=["lab", "legal", "research", "poc_template"],
)
def test_reference_hosts_enable_harness_memory_profile(env: ApplicationEnvironmentProfile) -> None:
    assert env.memory_profile.enable_task_memory is True
    assert env.memory_profile.enable_user_memory is True
    assert env.memory_profile.enable_org_memory is True
    assert env.memory_profile.enable_long_term_memory is True


def test_reference_hosts_materialize_runtime_config_enables_memory(tmp_path: Path) -> None:
    env = build_lab_environment_profile(LabApplicationSettings.from_env())
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    harness = LabHarnessContext(policy_bundle=RuntimePolicyBundle(), strict_harness=False)
    config = materialize_runtime_config(
        _request(),
        harness,
        env,
        llm_adapter=FakeLLMAdapter(),
    )
    assert config.enable_task_memory is True
    assert config.enable_user_profile_memory is True

    wiring = resolve_memory_platform_wiring(env)
    assert wiring.sqlite_bundle is not None
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)


def test_reference_hosts_sqlite_memory_wiring(tmp_path: Path) -> None:
    env = build_legal_environment_profile(LegalBackendSettings.from_env())
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    wiring = resolve_memory_platform_wiring(env)
    assert wiring.sqlite_bundle is not None
    assert isinstance(wiring.session_storage, SQLiteSessionStorage)
    assert isinstance(wiring.user_profile_store, SQLiteUserProfileStore)

    config = materialize_runtime_config(
        _request(),
        default_reference_harness(),
        env,
        llm_adapter=FakeLLMAdapter(),
    )
    assert config.enable_task_memory is True
    assert config.memory_scope_boundary == "tenant"
