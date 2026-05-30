# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from lab_application.host.agent_builders import LAB_AGENT_BUILDERS
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.wiring import build_lab_registry
from lab_application.manifest import build_lab_manifest

pytestmark = pytest.mark.unit


def test_build_lab_manifest_respects_echo_flag() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    manifest = build_lab_manifest(settings)
    paths = [b.import_path for b in manifest.enabled_agents()]
    assert "echo.echo_agent.EchoAgent" in paths
    assert not any("mock_agents" in p for p in paths)


def test_build_lab_registry_matches_legacy_flags() -> None:
    settings = LabApplicationSettings(
        include_echo=True,
        include_mock_agents=False,
        include_signoff_probe=False,
        include_research=False,
    )
    registry = build_lab_registry(settings=settings)
    assert registry.has("echo")
    assert len(registry.list_agent_ids()) == 1


def test_build_lab_registry_via_manifest_and_builders() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    manifest = build_lab_manifest(settings)
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
    registry = build_application_registry(manifest, ctx, builders=LAB_AGENT_BUILDERS)
    assert registry.has("echo")
