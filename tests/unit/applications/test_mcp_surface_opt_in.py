# © Artur Czarnecki. All rights reserved.

"""MCP must remain supported but opt-in per Tier-3 application.

MCP-RC MRC-5 (R-P7-02): the factory-level checks use the lab application as the vehicle. The previous vehicle,
the strict PRODUCT local-workspace app, cannot be built in a plain test environment any more (strict agent-contract
authority, memory admission evidence and tenant authority are all required), which has nothing to do with the MCP
opt-in behaviour under test. The lab app builds with an explicit provider and mounts ``/mcp`` when asked.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from intergrax.applications._shared.mcp_import_guard import MCPDependencyError, ensure_mcp_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]
APPLICATION_FACTORIES = (
    "governed_contractor_application",
    "lab_application",
    "legal_application",
    "local_workspace_application",
    "poc_template_application",
    "research_application",
    "dispute_sim_application",
    "intergrax_assistant_application",
)


@pytest.fixture(autouse=True)
def _explicit_provider_and_optional_sdk_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lab factory materializes an LLM adapter at build time: explicit provider, optional SDK stubbed.

    R-P7-02 root cause: the tests assumed an implicit provider and an installed optional ``ollama`` extra. A stub
    module stands in only when the real SDK is absent; nothing here performs an LLM call.
    """
    if importlib.util.find_spec("ollama") is None:
        stub = types.ModuleType("ollama")
        stub.Client = lambda *_args, **_kwargs: object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ollama", stub)
    monkeypatch.setenv("INTERGRAX_LLM_PROVIDER", "ollama")


def _module_level_mcp_imports(path: Path) -> list[str]:
    """Top-level (not function-local, not TYPE_CHECKING) imports of the optional MCP runtime."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            if module in {"fastmcp", "mcp", "intergrax.applications._shared.fastapi_mcp"} or module.startswith(
                ("fastmcp.", "mcp.")
            ):
                found.append(module)
    return found


@pytest.mark.parametrize("app_pkg", APPLICATION_FACTORIES)
def test_application_factory_has_no_module_level_mcp_imports(app_pkg: str) -> None:
    factory_path = REPO / "applications" / app_pkg / "host" / "factory.py"
    assert _module_level_mcp_imports(factory_path) == []


def _run_clean_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    import os

    env = {**os.environ, "INTERGRAX_LLM_PROVIDER": "ollama", "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, "-c", code, str(REPO)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=env,
    )


def test_mcp_import_guard_import_does_not_load_fastmcp() -> None:
    """Behavioural: importing the guard in a clean interpreter must not import fastmcp or the coupling."""
    completed = _run_clean_interpreter(
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import intergrax.applications._shared.mcp_import_guard;"
        "assert 'fastmcp' not in sys.modules, 'fastmcp imported';"
        "assert 'intergrax.applications._shared.fastapi_mcp' not in sys.modules, 'coupling imported'"
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.no_ci
def test_http_only_factory_does_not_import_fastmcp() -> None:
    """Behavioural: building the lab app without MCP in a clean interpreter never imports fastmcp."""
    completed = _run_clean_interpreter(
        "import sys, types, dataclasses\n"
        "root = sys.argv[1]\n"
        "sys.path[:0] = [root, root + '/applications', root + '/agents']\n"
        "try:\n"
        "    import ollama\n"
        "except ImportError:\n"
        "    stub = types.ModuleType('ollama'); stub.Client = lambda *a, **k: object(); sys.modules['ollama'] = stub\n"
        "from lab_application.host.settings import LabApplicationSettings\n"
        "from lab_application.host.factory import create_lab_application\n"
        "settings = dataclasses.replace(LabApplicationSettings.from_env(), include_mcp=False)\n"
        "create_lab_application(settings=settings)\n"
        "assert 'fastmcp' not in sys.modules, 'fastmcp imported by the HTTP-only factory'\n"
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


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
    from lab_application.host.factory import create_lab_application
    from lab_application.host.settings import LabApplicationSettings

    settings = dataclasses.replace(LabApplicationSettings.from_env(), include_mcp=True)
    with patch(
        "intergrax.applications._shared.mcp_import_guard.ensure_mcp_dependencies",
        side_effect=MCPDependencyError("INCLUDE_MCP=true"),
    ):
        with pytest.raises(MCPDependencyError, match="INCLUDE_MCP=true"):
            create_lab_application(settings=settings)


@pytest.mark.parametrize(
    "profile",
    [
        "lab",
        pytest.param(
            "product",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "TRANSFERRED (MCP-RC R-P7-02, not MCP-owned): the product scaffold fails STRICT agent-contract "
                    "authority resolution in package emission; strict=True forces removal once that is fixed"
                ),
            ),
        ),
    ],
)
def test_scaffold_defaults_mcp_to_opt_in(tmp_path: Path, profile: str) -> None:
    from tests.unit.applications.scaffold_runtime_helper import prepare_scaffold_package

    target, _pkg, _short = prepare_scaffold_package(
        tmp_path,
        name="mcp_opt_in_test",
        profile=profile,
        port=8298,
        route_prefix="/v1/mcp_opt_in_test",
        agents=["echo"],
    )
    env_text = (target / ".env.example").read_text(encoding="utf-8")
    assert "INCLUDE_MCP=false" in env_text
