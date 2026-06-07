# © Artur Czarnecki. All rights reserved.

"""REG-1/2: Harness registry snapshot and assembly validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.registry_assembly_resolver import (
    RegistryAssemblyError,
    assert_registry_assembly_valid,
    validate_registry_snapshot,
)
from intergrax.applications._shared.registry_snapshot import resolve_registry_snapshot
from intergrax.applications._shared.registry_wiring import resolve_registry_snapshot_protocol
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.runtime_bindings import SessionStorageBinding
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_registry_snapshot_from_build_context() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.snapshot")
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    snapshot = resolve_registry_snapshot(wiring.build_context)

    assert snapshot.integration_profile is not None
    assert snapshot.policy_bundle is not None
    assert snapshot.skill_registry is not None
    assert snapshot.skill_ids()


def test_resolve_registry_snapshot_protocol_returns_snapshot() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.protocol")
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    protocol = resolve_registry_snapshot_protocol(wiring.build_context)

    assert protocol.tool_ids() == wiring.registry_snapshot.tool_ids()  # type: ignore[union-attr]
    assert protocol.skill_ids() == wiring.registry_snapshot.skill_ids()  # type: ignore[union-attr]


def test_wire_application_environment_includes_registry_snapshot() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.wire")
    wiring = wire_application_environment(build_lab_manifest(settings), env)

    assert wiring.registry_snapshot is not None


def test_wire_application_environment_wires_session_storage_binding() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.session.binding")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    assert isinstance(wiring.tool_wiring.wiring_context.session_storage, SessionStorageBinding)
    assert wiring.registry_snapshot.policy_bundle is wiring.policy_bundle


def test_validate_registry_snapshot_requires_policy_bundle() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.policy")
    snapshot = resolve_registry_snapshot(
        ApplicationBuildContext.for_manifest(
            build_lab_manifest(LabApplicationSettings.from_env()),
            policy_bundle=None,
        ),
    )
    result = validate_registry_snapshot(snapshot, env)
    assert not result.valid
    assert any("policy_bundle" in error for error in result.errors)


def test_assert_registry_assembly_valid_raises_when_tool_registry_missing() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.tools")
    env.tool_profile = env.tool_profile.model_copy(
        update={"enabled_bundles": ["jira"]},
    )
    snapshot = resolve_registry_snapshot(
        ApplicationBuildContext.for_manifest(
            build_lab_manifest(LabApplicationSettings.from_env()),
            tool_registry=None,
            policy_bundle=RuntimePolicyBundle(),
            integration_profile=env.integration_profile,
        ),
    )
    with pytest.raises(RegistryAssemblyError, match="tool_registry"):
        assert_registry_assembly_valid(snapshot, env)


def test_assert_registry_assembly_valid_accepts_empty_tool_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.empty-tools")
    env.tool_profile = env.tool_profile.model_copy(
        update={"enabled": [], "enabled_bundles": [], "register_all_catalog_bundles": False},
    )
    env.skill_profile = env.skill_profile.model_copy(
        update={"enabled": [], "enabled_bundles": [], "register_all_catalog_bundles": False},
    )
    env.prompt_profile = env.prompt_profile.model_copy(update={"load_on_startup": False})
    snapshot = resolve_registry_snapshot(
        ApplicationBuildContext.for_manifest(
            build_lab_manifest(LabApplicationSettings.from_env()),
            tool_registry=None,
            skill_registry=None,
            policy_bundle=RuntimePolicyBundle(),
            integration_profile=env.integration_profile,
        ),
    )
    assert_registry_assembly_valid(snapshot, env)


def test_assert_registry_assembly_valid_requires_non_empty_tool_registry_when_enabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="reg.tool-ids")
    env.tool_profile = env.tool_profile.model_copy(
        update={"enabled_bundles": ["harness"]},
    )
    empty_registry = ToolRegistry()
    snapshot = resolve_registry_snapshot(
        ApplicationBuildContext.for_manifest(
            build_lab_manifest(LabApplicationSettings.from_env()),
            tool_registry=empty_registry,
            skill_registry=SkillRegistry(),
            policy_bundle=RuntimePolicyBundle(),
            integration_profile=env.integration_profile,
        ),
    )
    with pytest.raises(RegistryAssemblyError, match="tool_registry is empty"):
        assert_registry_assembly_valid(snapshot, env)
