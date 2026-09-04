# © Artur Czarnecki. All rights reserved.

"""NPSC-2: scaffold generators emit canonical harness host execution surfaces."""

from __future__ import annotations

import compileall
import importlib
import sys
from pathlib import Path

import pytest

from intergrax.scaffold.new_application import create_application

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _relax_harness_environment_assertions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.package_wiring.assert_manifest_package_closure",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.environment_wiring.assert_application_owned_tool_conformance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.diagnostic_assembly_resolver.assert_diagnostic_assembly_valid",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.environment_wiring.assert_skill_tool_requirements_for_profile",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.runtime.nexus.nexus_loop.validate_durable_attempt_lifecycle_for_composition",
        lambda **_kwargs: None,
    )

_FORBIDDEN_GENERATED_TOKENS = (
    "runtime.nexus_loop",
    ".execution.nexus_loop",
    "host_execution.nexus_loop",
    "resolve_harness_host_nexus_loop_legacy",
    "build_environment_host_task_execution",
    "build_host_task_execution",
    "harness_host_runtime_compat",
    "HarnessHostLegacyComposition",
    "from intergrax.runtime.nexus.nexus_loop import NexusLoop",
)

_SCAFFOLD_GENERATOR_PATHS = (
    Path("intergrax/scaffold/new_application.py"),
    Path("intergrax/scaffold/new_application_product.py"),
    Path("intergrax/scaffold/canonical_host_templates.py"),
)


def _iter_generated_py_files(app_root: Path) -> list[Path]:
    return sorted(path for path in app_root.rglob("*.py") if path.is_file())


def _assert_no_forbidden_tokens(text: str, *, label: str) -> None:
    for token in _FORBIDDEN_GENERATED_TOKENS:
        assert token not in text, f"{label} must not contain forbidden token: {token}"


def _assert_canonical_factory(factory_text: str) -> None:
    assert "host_execution = runtime.execution" in factory_text
    assert "mount_" in factory_text
    assert "host_execution=host_execution" in factory_text
    assert "registry=" in factory_text
    _assert_no_forbidden_tokens(factory_text, label="factory")


def _prepare_package(
    tmp_path: Path,
    *,
    name: str,
    profile: str,
    port: int,
    route_prefix: str,
) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "applications").mkdir(parents=True, exist_ok=True)
    target = create_application(
        name=name,
        agents=["echo"],
        profile=profile,
        root=root,
        port=port,
        route_prefix=route_prefix,
        force=True,
    )
    pkg = target.name
    short = pkg.removesuffix("_application")
    apps_dir = str(target.parent)
    if apps_dir not in sys.path:
        sys.path.insert(0, apps_dir)
    for key in list(sys.modules):
        if key == pkg or key.startswith(f"{pkg}."):
            del sys.modules[key]
    return target, pkg, short


def _purge_package(pkg: str) -> None:
    for key in list(sys.modules):
        if key == pkg or key.startswith(f"{pkg}."):
            del sys.modules[key]


@pytest.mark.parametrize("profile", ["lab", "product"])
def test_scaffold_generators_emit_canonical_execution_surface(tmp_path: Path, profile: str) -> None:
    port = 8291 if profile == "lab" else 8292
    route_prefix = f"/v1/npsc2_{profile}"
    target, pkg, short = _prepare_package(
        tmp_path,
        name=f"npsc2_{profile}",
        profile=profile,
        port=port,
        route_prefix=route_prefix,
    )

    generated_files = _iter_generated_py_files(target)
    assert generated_files

    combined = "\n".join(path.read_text(encoding="utf-8") for path in generated_files)
    _assert_no_forbidden_tokens(combined, label=f"generated {profile} application")

    factory_text = (target / "host" / "factory.py").read_text(encoding="utf-8")
    _assert_canonical_factory(factory_text)

    router_text = (target / "serving" / "fastapi_router.py").read_text(encoding="utf-8")
    assert "HostTaskExecutionPort" in router_text
    assert "AgentRegistryRead" in router_text
    assert "host_execution.execute" in router_text
    _assert_no_forbidden_tokens(router_text, label="serving router")

    mcp_text = (target / "mcp" / "server.py").read_text(encoding="utf-8")
    assert "host_execution: HostTaskExecutionPort" in mcp_text
    assert "registry: AgentRegistryRead" in mcp_text
    _assert_no_forbidden_tokens(mcp_text, label="mcp server")

    assert compileall.compile_dir(str(target), quiet=1)

    _purge_package(pkg)
    factory_mod = importlib.import_module(f"{pkg}.host.factory")
    factory_name = f"create_{short}_backend_app" if profile == "product" else f"create_{short}_application"
    create_app = factory_mod.__dict__[factory_name]
    assert callable(create_app)

    if profile == "lab":
        app = create_app()
        assert app is not None

    _purge_package(pkg)
    if str(target.parent) in sys.path:
        sys.path.remove(str(target.parent))


def test_scaffold_template_sources_gate_forbidden_execution_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for rel in _SCAFFOLD_GENERATOR_PATHS:
        text = (repo_root / rel).read_text(encoding="utf-8")
        for token in (
            "runtime.nexus_loop",
            "host_execution.nexus_loop",
            "resolve_harness_host_nexus_loop_legacy",
            "build_environment_host_task_execution",
        ):
            assert token not in text, f"{rel} must not reference forbidden token: {token}"
