# © Artur Czarnecki. All rights reserved.

"""NPSC-2: scaffold generators emit canonical harness host execution surfaces."""

from __future__ import annotations

import ast
import compileall
import importlib
import sys
from pathlib import Path

import pytest

from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.canonical_host_templates import (
    render_canonical_lab_serving_router_py,
    render_canonical_product_serving_router_py,
)
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


def _count_broad_except_handlers(source: str) -> tuple[int, int]:
    tree = ast.parse(source)
    exception_handlers = 0
    base_exception_handlers = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        handler_type = node.type
        if handler_type is None:
            exception_handlers += 1
            continue
        names: list[str] = []
        if isinstance(handler_type, ast.Name):
            names.append(handler_type.id)
        elif isinstance(handler_type, ast.Tuple):
            names.extend(
                elt.id for elt in handler_type.elts if isinstance(elt, ast.Name)
            )
        for name in names:
            if name == "Exception":
                exception_handlers += 1
            elif name == "BaseException":
                base_exception_handlers += 1
    return exception_handlers, base_exception_handlers


def _assert_no_broad_except_handlers(source: str, *, label: str) -> None:
    exception_handlers, base_exception_handlers = _count_broad_except_handlers(source)
    assert exception_handlers == 0, (
        f"{label} must not contain broad Exception handlers "
        f"(found {exception_handlers})"
    )
    assert base_exception_handlers == 0, (
        f"{label} must not contain BaseException handlers "
        f"(found {base_exception_handlers})"
    )


def _assert_run_agent_propagates_exceptions(router_source: str, *, label: str) -> None:
    tree = ast.parse(router_source)
    run_agent: ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_agent":
            run_agent = node
            break
    assert run_agent is not None, f"{label} must define run_agent route handler"
    for child in ast.walk(run_agent):
        assert not isinstance(child, ast.Try), (
            f"{label} run_agent must not locally catch exceptions"
        )


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
    for path in generated_files:
        source = path.read_text(encoding="utf-8")
        _assert_no_broad_except_handlers(
            source,
            label=f"generated {profile} file {path.relative_to(target)}",
        )

    factory_text = (target / "host" / "factory.py").read_text(encoding="utf-8")
    _assert_canonical_factory(factory_text)

    router_text = (target / "serving" / "fastapi_router.py").read_text(encoding="utf-8")
    assert "HostTaskExecutionPort" in router_text
    assert "AgentRegistryRead" in router_text
    assert "host_execution.execute" in router_text
    _assert_no_forbidden_tokens(router_text, label="serving router")
    _assert_no_broad_except_handlers(router_text, label=f"{profile} serving router")
    _assert_run_agent_propagates_exceptions(router_text, label=f"{profile} serving router")
    assert "HTTPException" not in router_text
    assert "HTTP_502_BAD_GATEWAY" not in router_text

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


def test_canonical_template_sources_do_not_emit_broad_except_handlers() -> None:
    names = ScaffoldApplicationNames.resolve(
        "npsc2q_lab",
        port=8291,
        route_prefix="/v1/npsc2q",
    )
    lab_router = render_canonical_lab_serving_router_py(names)
    product_router = render_canonical_product_serving_router_py(names, specs=[])
    for label, source in (
        ("lab template", lab_router),
        ("product template", product_router),
    ):
        _assert_no_broad_except_handlers(source, label=label)
        _assert_run_agent_propagates_exceptions(source, label=label)
        assert "except Exception" not in source
        assert "HTTP_502_BAD_GATEWAY" not in source


def test_canonical_template_module_has_no_broad_except_handlers() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    template_source = (repo_root / "intergrax/scaffold/canonical_host_templates.py").read_text(
        encoding="utf-8"
    )
    _assert_no_broad_except_handlers(template_source, label="canonical_host_templates.py")
