# © Artur Czarnecki. All rights reserved.

"""W-B Part D — LKW reference production on the hosted runner, loopback defaults (NT-WB-14..17, 25)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from intergrax.applications._shared.production_process_composition import ProductionProcessComposition

from local_workspace_application.host import main as lkw_main
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting.profile import build_local_workspace_hosted_profile
from research_application.host.settings import ResearchBackendSettings

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LKW_MAIN = _REPO_ROOT / "applications/local_workspace_application/host/main.py"
_RESEARCH_MAIN = _REPO_ROOT / "applications/research_application/host/main.py"

# Declared direct ``uvicorn.run`` sites after W-B Part D (record, not a standing gate: PT-G01).
_DECLARED_DIRECT_UVICORN_SITES = frozenset(
    {
        "applications/attestation_demo/host/main.py",
        "applications/governed_contractor_application/host/package_b_strict_production_serving_command.py",
        "applications/intergrax_assistant_application/host/main.py",
        "applications/lab_application/host/main.py",
        "applications/poc_template_application/host/main.py",
        "applications/research_application/host/main.py",
        "intergrax/cli/run.py",
        "intergrax/harness/app.py",
        "intergrax/scaffold/external_project/app.py",
    },
)


@dataclass(frozen=True)
class _HostedCall:
    process_composition: ProductionProcessComposition
    settings: LocalWorkspaceBackendSettings | None


def _is_uvicorn_run(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "uvicorn"
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _env_default(function: ast.FunctionDef, variable: str) -> str:
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == variable
        ):
            default = node.args[1]
            assert isinstance(default, ast.Constant)
            return str(default.value)
    raise AssertionError(f"{variable} default not found")


def _run_reference_production_with_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[_HostedCall], list[str]]:
    """Run ``run_reference_production`` with the lifecycle seams stubbed and both serving paths observed."""
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-wb-serving-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_SCHEDULER", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_INTERACTIONS", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    hosted_calls: list[_HostedCall] = []
    direct_calls: list[str] = []
    composition = SimpleNamespace(agent_platform_runtime=SimpleNamespace(stores=object()))

    class _Launcher:
        def deploy_and_activate(self, *_args: object, **_kwargs: object) -> None:
            return None

    def _capture_hosted(
        *,
        process_composition: ProductionProcessComposition,
        settings: LocalWorkspaceBackendSettings | None = None,
        **_kwargs: object,
    ) -> None:
        hosted_calls.append(_HostedCall(process_composition=process_composition, settings=settings))

    def _capture_direct(*_args: object, **_kwargs: object) -> None:
        direct_calls.append("uvicorn.run")

    monkeypatch.setattr(lkw_main, "create_reference_production_process_composition", lambda: composition)
    monkeypatch.setattr(
        lkw_main,
        "build_local_workspace_reference_lifecycle_input",
        lambda: (
            SimpleNamespace(runtime_revision=SimpleNamespace(runtime_revision_id="wb-revision")),
            SimpleNamespace(artifact_locator=None),
        ),
    )
    monkeypatch.setattr(
        lkw_main,
        "wire_governed_reference_production_launcher",
        lambda _composition, _env: (_Launcher(), SimpleNamespace(principal=None)),
    )
    monkeypatch.setattr(lkw_main, "prepare_reference_runtime_materialization", lambda *_a, **_k: None)
    monkeypatch.setattr(lkw_main, "run_local_workspace_hosted_application", _capture_hosted)
    monkeypatch.setattr("uvicorn.run", _capture_direct)
    lkw_main.run_reference_production()
    return hosted_calls, direct_calls


def test_nt_wb_14_lkw_reference_production_hosted_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_WORKSPACE_BACKEND_HOST", raising=False)
    hosted_calls, direct_calls = _run_reference_production_with_capture(monkeypatch)
    assert direct_calls == []
    assert len(hosted_calls) == 1
    profile = build_local_workspace_hosted_profile(
        process_composition=hosted_calls[0].process_composition,
        settings=hosted_calls[0].settings,
    )
    assert getattr(profile.application_factory, "bind_host") == "127.0.0.1"


def test_nt_wb_15_lkw_reference_production_has_no_uvicorn_run() -> None:
    tree = ast.parse(_LKW_MAIN.read_text(encoding="utf-8"))
    function = _function(tree, "run_reference_production")
    assert not any(_is_uvicorn_run(node) for node in ast.walk(function))


def test_nt_wb_16_research_default_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_BACKEND_HOST", raising=False)
    assert ResearchBackendSettings().backend_host == "127.0.0.1"
    tree = ast.parse(_RESEARCH_MAIN.read_text(encoding="utf-8"))
    assert _env_default(_function(tree, "run_reference_production"), "RESEARCH_BACKEND_HOST") == "127.0.0.1"


def test_nt_wb_17_explicit_all_interfaces_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_BACKEND_HOST", "0.0.0.0")
    hosted_calls, direct_calls = _run_reference_production_with_capture(monkeypatch)
    assert direct_calls == []
    profile = build_local_workspace_hosted_profile(
        process_composition=hosted_calls[0].process_composition,
        settings=hosted_calls[0].settings,
    )
    assert getattr(profile.application_factory, "bind_host") == "0.0.0.0"


def test_nt_wb_25_uvicorn_run_site_count_record() -> None:
    sites: set[str] = set()
    for root in ("applications", "intergrax"):
        for path in (_REPO_ROOT / root).rglob("*.py"):
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if "/tests/" in relative or "/vendor_knowledge/" in relative:
                continue
            text = path.read_text(encoding="utf-8")
            if "uvicorn.run(" not in text:
                continue
            tree = ast.parse(text)
            if any(_is_uvicorn_run(node) for node in ast.walk(tree)):
                sites.add(relative)
    assert sites == _DECLARED_DIRECT_UVICORN_SITES
