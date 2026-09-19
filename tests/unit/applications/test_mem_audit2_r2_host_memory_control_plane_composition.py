# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-2-R2: host-owned single MemoryControlPlane composition guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.memory_wiring import build_session_manager_from_environment
from intergrax.applications._shared.runtime_config_bridge import (
    build_runtime_context_from_environment,
)
from intergrax.runtime.nexus.context.memory_context_invocation import (
    memory_control_plane_from_config,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter, build_runtime_request_for_tests
from testing_support.memory_control_plane_test_stub import MemoryControlPlaneTestStub

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_REPO = Path(__file__).resolve().parents[3]
_RUNTIME_BRIDGE = _REPO / "intergrax" / "applications" / "_shared" / "runtime_config_bridge.py"


def _runtime_request() -> RuntimeRequest:
    return build_runtime_request_for_tests(
        seed="mem-audit2-r2",
        tenant_id="tenant-mem-r2",
        agent_id="echo",
        user_id="user-mem-r2",
        session_id="session-mem-r2",
        message="memory control plane composition probe",
    )


def test_runtime_config_bridge_does_not_build_duplicate_memory_control_plane() -> None:
    source = _RUNTIME_BRIDGE.read_text(encoding="utf-8")
    assert "build_default_memory_control_plane" not in source


def test_build_runtime_context_shares_host_memory_control_plane(tmp_path: Path) -> None:
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        tenant_id="tenant-mem-r2",
    )
    runtime_ctx = build_runtime_context_from_environment(
        _runtime_request(),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )
    session_manager = runtime_ctx.session_manager
    assert session_manager is not None
    host_plane = session_manager.memory_control_plane
    assert host_plane is not None

    wiring = runtime_ctx.config.tool_wiring_context
    assert wiring is not None
    tool_plane = wiring.extras.get("memory_control_plane")
    assert tool_plane is host_plane
    assert memory_control_plane_from_config(runtime_ctx.config) is host_plane


def test_wire_application_environment_shares_host_memory_control_plane(
    tmp_path: Path,
) -> None:
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        tenant_id="tenant-mem-r2",
    )
    session_manager = wiring.composition.tool_wiring_context.extras["session_manager"]
    host_plane = session_manager.memory_control_plane
    assert host_plane is not None
    assert wiring.composition.tool_wiring_context.extras.get("memory_control_plane") is host_plane


def test_custom_memory_control_plane_injected_via_host_builder(tmp_path: Path) -> None:
    env = build_lab_environment_profile(LabApplicationSettings.from_env())
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    custom_plane = MemoryControlPlaneTestStub()
    session_manager = build_session_manager_from_environment(
        env,
        tenant_id="tenant-custom-plane",
        memory_control_plane=custom_plane,
    )
    assert session_manager.memory_control_plane is custom_plane
