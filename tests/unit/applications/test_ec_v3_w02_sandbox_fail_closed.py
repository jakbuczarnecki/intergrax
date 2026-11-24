# © Artur Czarnecki. All rights reserved.

"""EC-V3-W02 sandbox guarded composition regression (A4-032b, T-IGX-074-A)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from intergrax.applications._shared.sandbox_wiring import (
    SandboxCompositionError,
    product_requires_sandbox,
    sandbox_required_for_posture,
    wire_sandbox_sessions,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.sandbox_profile import SandboxProfile
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.providers.workspace.bundle import WORKSPACE_BUNDLE_ID
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import UnknownToolBundleError

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _with_posture(
    base: ApplicationEnvironmentProfile,
    *,
    profile: ApplicationProfile,
    mode: ExecutionMode,
) -> ApplicationEnvironmentProfile:
    meta = base.meta.model_copy(update={"application_profile": profile, "execution_mode": mode})
    return base.model_copy(update={"meta": meta})


@pytest.fixture
def _catalog() -> Iterator[None]:
    register_default_tools()
    yield
    reset_default_tools_bootstrap()


def test_sbx_01_product_missing_sandbox_profile() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["workspace.read_file"])
    with pytest.raises(SandboxCompositionError, match="SandboxProfile"):
        wire_sandbox_sessions(env)


def test_sbx_02_product_missing_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_SANDBOX_ROOT", raising=False)
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["sandbox.exec"])
    env = env.model_copy(update={"sandbox": SandboxProfile(enable_exec_tool=True)})
    with pytest.raises(SandboxCompositionError, match="explicit sandbox root"):
        wire_sandbox_sessions(env)
    assert not (tmp_path / "build").exists()


def test_sbx_03_lab_strict_missing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_SANDBOX_ROOT", raising=False)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="ec.w02.sbx.03")
    env = _with_posture(base, profile=ApplicationProfile.LAB, mode=ExecutionMode.STRICT)
    env = env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={"tools": ToolProfile(enabled=["workspace.write_file"])},
            ),
            "sandbox": SandboxProfile(enable_exec_tool=True),
        },
    )
    assert sandbox_required_for_posture(env) is True
    with pytest.raises(SandboxCompositionError):
        wire_sandbox_sessions(env)


def test_sbx_04_side_effect_only_via_bundle(_catalog: None) -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={"tools": ToolProfile(enabled_bundles=[WORKSPACE_BUNDLE_ID])},
            ),
        },
    )
    assert sandbox_required_for_posture(env) is True


def test_sbx_05_all_catalog_bundles(_catalog: None) -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={"tools": ToolProfile(register_all_catalog_bundles=True)},
            ),
        },
    )
    assert sandbox_required_for_posture(env) is True


def test_sbx_06_profile_root_used(tmp_path: Path) -> None:
    root = tmp_path / "profile-root"
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["gitlab.issue_create"])
    env = env.model_copy(update={"sandbox": SandboxProfile(root=root, enable_exec_tool=True)})
    manager = wire_sandbox_sessions(env)
    assert manager is not None
    assert manager.root == root


def test_sbx_07_environment_root_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "env-root"
    monkeypatch.setenv("INTERGRAX_SANDBOX_ROOT", str(env_root))
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["websearch.search"])
    env = env.model_copy(update={"sandbox": SandboxProfile(enable_exec_tool=True)})
    manager = wire_sandbox_sessions(env)
    assert manager is not None
    assert manager.root == env_root


def test_sbx_08_profile_root_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_root = tmp_path / "profile"
    env_root = tmp_path / "env"
    monkeypatch.setenv("INTERGRAX_SANDBOX_ROOT", str(env_root))
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["sandbox.exec"])
    env = env.model_copy(
        update={"sandbox": SandboxProfile(root=profile_root, enable_exec_tool=True)},
    )
    manager = wire_sandbox_sessions(env)
    assert manager is not None
    assert manager.root == profile_root


def test_sbx_09_whitespace_environment_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_SANDBOX_ROOT", "   ")
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["workspace.read_file"])
    env = env.model_copy(update={"sandbox": SandboxProfile(enable_exec_tool=True)})
    with pytest.raises(SandboxCompositionError):
        wire_sandbox_sessions(env)


def test_sbx_10_non_guarded_lab_implicit_root_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ec.w02.sbx.10")
    env = env.model_copy(update={"sandbox": SandboxProfile()})
    manager = wire_sandbox_sessions(env)
    assert manager is not None
    assert manager.root == Path("build") / "sandbox_sessions"


def test_sbx_11_guarded_no_side_effects_without_sandbox() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env.sandbox = None
    assert sandbox_required_for_posture(env) is False
    assert wire_sandbox_sessions(env) is None


def test_sbx_12_unresolved_bundle_not_false_negative() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={"tools": ToolProfile(enabled_bundles=["bundle.does-not-exist"])},
            ),
        },
    )
    with pytest.raises(UnknownToolBundleError):
        sandbox_required_for_posture(env)


def test_sbx_13_exec_tool_enablement_requires_explicit_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_SANDBOX_ROOT", raising=False)
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(update={"sandbox": SandboxProfile(enable_exec_tool=True)})
    with pytest.raises(SandboxCompositionError):
        wire_sandbox_sessions(env)
    env = env.model_copy(
        update={"sandbox": SandboxProfile(enable_exec_tool=True, root=tmp_path / "exec")},
    )
    manager = wire_sandbox_sessions(env)
    assert manager is not None


def test_sbx_14_product_predicate_unchanged_lab_strict() -> None:
    env = _with_posture(
        ApplicationEnvironmentProfile.lab_defaults(profile_id="ec.w02.sbx.14"),
        profile=ApplicationProfile.LAB,
        mode=ExecutionMode.STRICT,
    )
    env = env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={"tools": ToolProfile(enabled=["sandbox.exec"])},
            ),
        },
    )
    assert product_requires_sandbox(env) is False
    assert sandbox_required_for_posture(env) is True


def test_sbx_negative_does_not_construct_manager() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["workspace.write_file"])
    with patch(
        "intergrax.applications._shared.sandbox_wiring.SandboxSessionManager",
        MagicMock(),
    ) as manager_cls:
        with pytest.raises(SandboxCompositionError):
            wire_sandbox_sessions(env)
        manager_cls.assert_not_called()


def test_sbx_15_explicit_member_activates_side_effect_bundle(_catalog: None) -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={"tools": ToolProfile(enabled=["workspace.read_file"])},
            ),
        },
    )
    assert sandbox_required_for_posture(env) is True


def test_sbx_16_mixed_explicit_ids_and_enabled_bundles(_catalog: None, tmp_path: Path) -> None:
    from intergrax.tools.providers.sandbox.bundle import SANDBOX_BUNDLE_ID

    root = tmp_path / "mixed"
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={
                    "tools": ToolProfile(
                        enabled=["echo.ping"],
                        enabled_bundles=[SANDBOX_BUNDLE_ID],
                    ),
                },
            ),
            "sandbox": SandboxProfile(root=root, enable_exec_tool=True),
        },
    )
    assert sandbox_required_for_posture(env) is True
    manager = wire_sandbox_sessions(env)
    assert manager is not None
    assert manager.root == root


def test_sbx_17_integration_adds_side_effect_tools_after_expansion() -> None:
    from intergrax.applications._shared.integration_tool_profile import extend_tool_profile_for_integration
    from intergrax.integrations.registry.profile import IntegrationProfile

    profile = ToolProfile()
    integration = IntegrationProfile(search_provider="tavily")
    expanded = extend_tool_profile_for_integration(profile, integration)
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(update={"capabilities": env.capabilities.model_copy(update={"tools": expanded})})
    assert sandbox_required_for_posture(env) is True


def test_sbx_18_codecraft_expansion_adds_exec_side_effect_tools() -> None:
    from intergrax.applications._shared.codecraft_wiring import tool_profile_with_codecraft
    from intergrax.applications._shared.sandbox_wiring import tool_profile_with_sandbox
    from intergrax.contracts.codecraft_profile import CodeCraftProfile

    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="ec.w02.sbx.18")
    env = _with_posture(base, profile=ApplicationProfile.LAB, mode=ExecutionMode.STRICT)
    env = env.model_copy(
        update={
            "codecraft_profile": CodeCraftProfile(mode="supervised"),
            "sandbox": SandboxProfile(enable_exec_tool=True),
        },
    )
    effective = tool_profile_with_sandbox(
        env.model_copy(update={"tool_profile": tool_profile_with_codecraft(env)}),
    )
    wiring_env = env.model_copy(
        update={"capabilities": env.capabilities.model_copy(update={"tools": effective})},
    )
    assert sandbox_required_for_posture(wiring_env) is True


def test_sbx_19_missing_root_after_effective_tool_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _catalog: None,
) -> None:
    from intergrax.applications._shared.environment_wiring import wire_application_environment
    from intergrax.applications.contracts.manifest import ApplicationManifest

    monkeypatch.delenv("INTERGRAX_SANDBOX_ROOT", raising=False)
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["workspace.read_file"])
    manifest = ApplicationManifest.lab(
        app_id="ec_w02_sbx_expansion",
        name="EC W02 Sandbox Expansion",
        route_prefix="/v1/ec_w02_sbx",
        env_prefix="EC_W02_SBX_",
        agents=[],
        environment=env,
    )
    with pytest.raises(SandboxCompositionError, match="explicit sandbox root"):
        wire_application_environment(manifest, env, conformance_check=False)


def test_sbx_20_composition_passes_effective_tool_profile_to_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _catalog: None,
) -> None:
    from intergrax.applications._shared import environment_wiring as environment_wiring_module
    from intergrax.applications._shared.environment_wiring import wire_application_environment
    from intergrax.applications.contracts.manifest import ApplicationManifest

    monkeypatch.setattr(
        "intergrax.applications._shared.runtime_event_delivery_wiring.require_otlp_observability_dependency_profile",
        lambda **_: None,
    )
    captured_profiles: list[ToolProfile] = []
    original_wire = environment_wiring_module.wire_sandbox_sessions

    def _capture_sandbox_env(env: ApplicationEnvironmentProfile) -> object:
        captured_profiles.append(env.tool_profile)
        return original_wire(env)

    monkeypatch.setattr(environment_wiring_module, "wire_sandbox_sessions", _capture_sandbox_env)
    root = tmp_path / "composition"
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["workspace.read_file"])
    env = env.model_copy(
        update={
            "sandbox": SandboxProfile(root=root, enable_exec_tool=True),
            "observability_profile": env.observability_profile.model_copy(
                update={"otlp_export_endpoint": "http://127.0.0.1:0/v1/logs"},
            ),
        },
    )
    manifest = ApplicationManifest.lab(
        app_id="ec_w02_sbx_compose",
        name="EC W02 Sandbox Compose",
        route_prefix="/v1/ec_w02_compose",
        env_prefix="EC_W02_COMPOSE_",
        agents=[],
        environment=env,
    )
    wiring = wire_application_environment(manifest, env, conformance_check=False)
    assert captured_profiles
    assert captured_profiles[-1] == wiring.tool_wiring.profile
    assert wiring.sandbox_manager is not None
    assert wiring.sandbox_manager.root == root
