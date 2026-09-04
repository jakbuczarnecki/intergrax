# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications.contracts.application_capability_projection import (
    application_capability_descriptor_from_manifest,
    resolve_binding_contract_id,
)
from intergrax.contracts.application_capability_metadata import (
    ApplicationCapabilityDescriptor,
    ApplicationCapabilityProjectionConflict,
    merge_application_capability_descriptors,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest


def test_manifest_projects_to_descriptor() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab",
        name="Lab",
        version="2.1.0",
        default_capability="echo.basic",
        agents=[
            AgentBinding.reference("echo"),
            AgentBinding.reference("legal", enabled=False),
        ],
    )
    descriptor = application_capability_descriptor_from_manifest(manifest)
    assert descriptor == ApplicationCapabilityDescriptor(
        application_id="lab",
        application_version="2.1.0",
        agent_contract_ids=("echo",),
        default_capability="echo.basic",
    )


def test_disabled_agent_excluded_from_projection() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab",
        name="Lab",
        agents=[
            AgentBinding.reference("echo"),
            AgentBinding.reference("legal", enabled=False),
        ],
    )
    descriptor = application_capability_descriptor_from_manifest(manifest)
    assert descriptor.agent_contract_ids == ("echo",)


def test_version_preserved_in_projection() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab",
        name="Lab",
        version="9.8.7",
        agents=[AgentBinding.reference("echo")],
    )
    descriptor = application_capability_descriptor_from_manifest(manifest)
    assert descriptor.application_version == "9.8.7"


def test_default_capability_preserved_in_projection() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab",
        name="Lab",
        default_capability="echo.basic",
        agents=[AgentBinding.reference("echo")],
    )
    descriptor = application_capability_descriptor_from_manifest(manifest)
    assert descriptor.default_capability == "echo.basic"


def test_unresolved_enabled_binding_identity_fails_closed() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab",
        name="Lab",
        agents=[AgentBinding(import_path="echo.echo_agent.EchoAgent")],
    )
    with pytest.raises(ApplicationCapabilityProjectionConflict, match="contract_id"):
        application_capability_descriptor_from_manifest(manifest)


def test_resolve_binding_contract_id_returns_declarative_contract_id() -> None:
    binding = AgentBinding.reference("agent.foo")
    assert resolve_binding_contract_id(binding) == "agent.foo"


def test_resolve_binding_contract_id_requires_declarative_identity() -> None:
    binding = AgentBinding(import_path="echo.echo_agent.EchoAgent")
    with pytest.raises(ApplicationCapabilityProjectionConflict):
        resolve_binding_contract_id(binding)


def test_resolve_binding_contract_id_does_not_instantiate_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []
    original = AgentBinding.resolved_agent_type

    def _track_resolved_agent_type(self: AgentBinding) -> type[object]:
        called.append(True)
        return original(self)

    monkeypatch.setattr(AgentBinding, "resolved_agent_type", _track_resolved_agent_type)

    binding = AgentBinding(import_path="echo.echo_agent.EchoAgent", contract_id="agent.foo")
    assert resolve_binding_contract_id(binding) == "agent.foo"
    assert called == []

    binding_without_id = AgentBinding(import_path="echo.echo_agent.EchoAgent")
    with pytest.raises(ApplicationCapabilityProjectionConflict):
        resolve_binding_contract_id(binding_without_id)
    assert called == []


def test_merge_application_capability_descriptors_dedupes_identical_rows() -> None:
    descriptor = ApplicationCapabilityDescriptor(
        application_id="lab",
        application_version="1.0.0",
        agent_contract_ids=("echo",),
    )
    merged = merge_application_capability_descriptors((descriptor, descriptor))
    assert merged == (descriptor,)


def test_merge_application_capability_descriptors_raises_on_conflict() -> None:
    first = ApplicationCapabilityDescriptor(
        application_id="lab",
        application_version="1.0.0",
        agent_contract_ids=("echo",),
    )
    second = ApplicationCapabilityDescriptor(
        application_id="lab",
        application_version="2.0.0",
        agent_contract_ids=("echo",),
    )
    with pytest.raises(ApplicationCapabilityProjectionConflict, match="conflicting"):
        merge_application_capability_descriptors((first, second))
