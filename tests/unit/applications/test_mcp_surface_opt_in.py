# © Artur Czarnecki. All rights reserved.

"""MCP must remain supported but opt-in per Tier-3 application."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.applications._shared.mcp_import_guard import MCPDependencyError, ensure_mcp_dependencies
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]
_MCP_IMPORT_RE = re.compile(
    r"^(?:from|import)\s+(?:fastmcp|mcp(?:\.\w+)*|intergrax\.applications\._shared\.fastapi_mcp)\b"
)

APPLICATION_FACTORIES = (
    "lab_application",
    "legal_application",
    "local_workspace_application",
    "poc_template_application",
    "research_application",
    "dispute_sim_application",
    "intergrax_assistant_application",
)


def _http_only_settings(**overrides: object) -> LocalWorkspaceBackendSettings:
    base = {
        "environment": LocalWorkspaceBackendSettings.from_env().environment,
        "include_mcp": False,
        "include_scheduler": False,
        "include_task_control": False,
        "include_interaction_routes": False,
    }
    base.update(overrides)
    return LocalWorkspaceBackendSettings(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("app_pkg", APPLICATION_FACTORIES)
def test_application_factory_has_no_module_level_mcp_imports(app_pkg: str) -> None:
    factory_path = REPO / "applications" / app_pkg / "host" / "factory.py"
    for line in factory_path.read_text(encoding="utf-8").splitlines():
        if _MCP_IMPORT_RE.match(line.strip()) and not line.startswith((" ", "\t")):
            pytest.fail(f"{factory_path}: module-level MCP import: {line.strip()}")


def test_mcp_import_guard_module_has_no_fastmcp_dependency() -> None:
    text = (REPO / "intergrax" / "applications" / "_shared" / "mcp_import_guard.py").read_text(
        encoding="utf-8"
    )
    assert "fastapi_mcp" not in text.split("load_mcp_coupling", maxsplit=1)[0]
    assert "from fastmcp" not in text.split("def ensure_mcp_dependencies", maxsplit=1)[0]


@pytest.mark.no_ci
def test_http_only_factory_does_not_import_fastmcp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    imported: list[str] = []
    import builtins

    real_import = builtins.__import__

    def track_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", track_import)

    from local_workspace_application.host.factory import create_local_workspace_backend_app

    create_local_workspace_backend_app(settings=_http_only_settings())
    blocked = {"fastmcp", "mcp"}
    assert not any(
        module in blocked or module.startswith("fastmcp.") or module.startswith("mcp.")
        for module in imported
    )


@pytest.mark.no_ci
def test_mcp_enabled_raises_clear_error_when_fastmcp_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "fastmcp", raising=False)

    import builtins

    real_import = builtins.__import__

    def block_fastmcp(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastmcp":
            raise ImportError("simulated missing fastmcp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_fastmcp)
    with pytest.raises(MCPDependencyError, match="INCLUDE_MCP=true"):
        ensure_mcp_dependencies()


@pytest.mark.no_ci
def test_mcp_enabled_factory_raises_when_fastmcp_missing() -> None:
    from local_workspace_application.host.factory import create_local_workspace_backend_app

    settings = _http_only_settings(include_mcp=True)
    with patch(
        "intergrax.applications._shared.mcp_import_guard.ensure_mcp_dependencies",
        side_effect=MCPDependencyError("INCLUDE_MCP=true"),
    ):
        with pytest.raises(MCPDependencyError, match="INCLUDE_MCP=true"):
            create_local_workspace_backend_app(settings=settings)


def test_scaffold_defaults_mcp_to_opt_in(tmp_path: Path) -> None:
    from tests.unit.applications.scaffold_runtime_helper import (
        import_scaffold_modules,
        prepare_scaffold_package,
        product_settings_class,
    )

    target, pkg, short = prepare_scaffold_package(
        tmp_path,
        name="mcp_opt_in_test",
        profile="product",
        port=8298,
        route_prefix="/v1/mcp_opt_in_test",
        agents=["echo"],
    )
    settings_text = (target / "host" / "settings.py").read_text(encoding="utf-8")
    env_text = (target / ".env.example").read_text(encoding="utf-8")
    assert "IntergraxApplicationSettingsBase" in settings_text
    assert "INCLUDE_MCP=false" in env_text

    _, settings_mod = import_scaffold_modules(pkg)
    settings_cls = product_settings_class(settings_mod, short)
    assert settings_cls.from_env().include_mcp is False
