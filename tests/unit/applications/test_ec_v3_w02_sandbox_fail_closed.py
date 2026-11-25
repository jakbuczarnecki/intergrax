# © Artur Czarnecki. All rights reserved.

"""T-IGX-074-A (EC-V3-W02) package-gate sandbox qualification isolation and fail-closed posture."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.applications._shared.package_wiring import (
    build_application_package,
    package_gate_environment,
    package_gate_metadata_environment,
    package_gate_qualification,
    release_all_package_gate_qualifications,
)
from intergrax.applications._shared.sandbox_wiring import (
    SandboxCompositionError,
    product_requires_sandbox,
    wire_sandbox_sessions,
)
from intergrax.applications.contracts.application_host import ApplicationFeatures, ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.sandbox_profile import SandboxProfile
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TOOLS = ["sandbox.exec", "workspace.write_file"]


def _strict_product_manifest() -> ApplicationManifest:
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=_TOOLS)
    env = env.model_copy(
        update={
            "sandbox": SandboxProfile(enable_exec_tool=True),
            "observability_profile": env.observability_profile.model_copy(
                update={"otlp_export_endpoint": ""},
            ),
        },
    )
    return ApplicationManifest(
        app_id="ecv3_gate_host",
        name="ECV3 Gate Host",
        description="",
        version="0.1.0",
        profile=ApplicationProfile.PRODUCT,
        route_prefix="/v1/ecv3",
        env_prefix="ECV3_",
        default_host="127.0.0.1",
        default_port=8090,
        default_capability=None,
        agents=[],
        integration_profile=IntegrationProfile.lab(),
        features=ApplicationFeatures.lab_defaults(),
        environment=env,
    )


def test_package_gate_qualification_runs_use_isolated_sandbox_roots() -> None:
    manifest = _strict_product_manifest()
    source = manifest.resolved_environment()
    with package_gate_qualification(source) as first:
        with package_gate_qualification(source) as second:
            first_root = first.composition_env.sandbox
            second_root = second.composition_env.sandbox
            assert first_root is not None and second_root is not None
            assert first_root.root != second_root.root
            assert first_root.root is not None
            assert first_root.root.is_dir()
            assert first_root.root.name.startswith("igx-package-gate-sandbox-")
    release_all_package_gate_qualifications()


def test_package_gate_qualification_cleans_up_on_success(tmp_path: Path) -> None:
    manifest = _strict_product_manifest()
    source = manifest.resolved_environment()
    with package_gate_qualification(source) as run:
        root = run.composition_env.sandbox
        assert root is not None and root.root is not None
        tracked = root.root
        assert tracked.is_dir()
    assert not tracked.exists()


def test_package_gate_qualification_cleans_up_on_exception(tmp_path: Path) -> None:
    manifest = _strict_product_manifest()
    source = manifest.resolved_environment()
    tracked: Path | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with package_gate_qualification(source) as run:
            sandbox = run.composition_env.sandbox
            assert sandbox is not None and sandbox.root is not None
            tracked = sandbox.root
            raise RuntimeError("boom")
    assert tracked is not None
    assert not tracked.exists()


def test_package_gate_metadata_preserves_source_sandbox_policy() -> None:
    explicit_root = Path("/tmp/explicit-sandbox-root")
    source = ApplicationEnvironmentProfile.product_defaults(tool_ids=_TOOLS)
    source = source.model_copy(
        update={
            "sandbox": SandboxProfile(root=explicit_root, enable_exec_tool=False),
        },
    )
    metadata = package_gate_metadata_environment(source)
    assert metadata.sandbox == source.sandbox
    gate = package_gate_environment(source)
    assert gate.sandbox is not None
    assert gate.sandbox.root == explicit_root
    assert gate.sandbox.enable_exec_tool is False


def test_package_gate_metadata_yields_stable_package_checksum() -> None:
    manifest = _strict_product_manifest()
    source = manifest.resolved_environment()
    with package_gate_qualification(source) as first:
        with package_gate_qualification(source) as second:
            pkg_a = build_application_package(manifest, first.metadata_env)
            pkg_b = build_application_package(manifest, second.metadata_env)
    assert pkg_a.distribution.checksum == pkg_b.distribution.checksum
    assert first.metadata_env.sandbox is not None
    assert first.metadata_env.sandbox.root is None


def test_product_requires_sandbox_without_qualification_fallback() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(tool_ids=["sandbox.exec"])
    assert product_requires_sandbox(env)
    with pytest.raises(SandboxCompositionError):
        wire_sandbox_sessions(env.model_copy(update={"sandbox": None}))
