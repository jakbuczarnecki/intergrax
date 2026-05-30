# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.wiring import (
    AgentImportError,
    ApplicationManifestConformanceError,
    build_application_registry,
    build_registry_from_manifest,
    load_agent_from_binding,
    validate_manifest_wiring,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = pytest.mark.unit


def test_load_agent_from_binding_echo() -> None:
    binding = AgentBinding.mount(EchoAgent)
    agent = load_agent_from_binding(binding)
    assert agent.get_contract().id == "echo"
    assert "echo.basic" in agent.get_contract().capabilities


def test_build_registry_from_manifest_echo() -> None:
    manifest = ApplicationManifest.lab(
        app_id="test_lab",
        name="Test Lab",
        agents=[AgentBinding.mount(EchoAgent)],
    )
    registry = build_registry_from_manifest(manifest)
    assert registry.has("echo")
    assert registry.find_by_capability("echo.basic")


def test_build_registry_contract_id_override() -> None:
    manifest = ApplicationManifest.lab(
        app_id="override_lab",
        name="Override",
        agents=[
            AgentBinding.mount(
                EchoAgent,
                contract_id="echo-lab",
            )
        ],
    )
    registry = build_registry_from_manifest(manifest)
    assert registry.has("echo-lab")
    assert not registry.has("echo")


def test_build_registry_skips_disabled_agents() -> None:
    manifest = ApplicationManifest.lab(
        app_id="partial",
        name="Partial",
        agents=[AgentBinding.mount(EchoAgent, enabled=False)],
    )
    with pytest.raises(ApplicationManifestConformanceError, match="no enabled agents"):
        build_registry_from_manifest(manifest)


def test_build_registry_via_type_keyed_builder() -> None:
    manifest = ApplicationManifest.lab(
        app_id="typed_lab",
        name="Typed",
        agents=[AgentBinding.mount(EchoAgent)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)

    def echo_builder(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> EchoAgent:
        return EchoAgent()

    registry = build_application_registry(
        manifest,
        ctx,
        builders={EchoAgent: echo_builder},
    )
    assert registry.has("echo")


def test_build_registry_via_typed_factory_on_binding() -> None:
    manifest = ApplicationManifest.lab(
        app_id="factory_lab",
        name="Factory",
        agents=[
            AgentBinding.mount(
                EchoAgent,
                factory=lambda _ctx, _b: EchoAgent(),
            )
        ],
    )
    registry = build_registry_from_manifest(manifest)
    assert registry.has("echo")


def test_agent_binding_mount_rejects_factory_and_builder_key() -> None:
    with pytest.raises(ValueError, match="factory or builder_key"):
        AgentBinding.mount(
            EchoAgent,
            factory=lambda _c, _b: EchoAgent(),
            builder_key="echo",
        )


def test_validate_manifest_duplicate_contract_ids_enabled() -> None:
    manifest = ApplicationManifest.lab(
        app_id="dup2",
        name="Dup2",
        agents=[
            AgentBinding.mount(EchoAgent, contract_id="same-id"),
            AgentBinding.mount(EchoAgent, contract_id="same-id"),
        ],
    )
    errors = validate_manifest_wiring(manifest)
    assert any("duplicate contract_id" in e for e in errors)


def test_load_agent_import_error_deserialize() -> None:
    with pytest.raises(AgentImportError):
        load_agent_from_binding(
            AgentBinding.deserialize(import_path="no_such_pkg.no_module.NoAgent"),
        )


def test_build_registry_duplicate_native_ids_raises() -> None:
    manifest = ApplicationManifest.lab(
        app_id="twice",
        name="Twice",
        agents=[
            AgentBinding.mount(EchoAgent),
            AgentBinding.mount(EchoAgent),
        ],
    )
    with pytest.raises(ValueError, match="already registered"):
        build_registry_from_manifest(manifest)


def test_deserialize_string_binding_still_works() -> None:
    binding = AgentBinding.deserialize(import_path="echo.echo_agent.EchoAgent")
    assert binding.agent_type is None
    assert binding.resolved_agent_type() is EchoAgent
    registry = build_registry_from_manifest(
        ApplicationManifest.lab(app_id="ser", name="Ser", agents=[binding])
    )
    assert registry.has("echo")
