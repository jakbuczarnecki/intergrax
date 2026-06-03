# © Artur Czarnecki. All rights reserved.

"""Phase N.9 — full scaffold acceptance (lab + product profiles, runtime E2E)."""

from __future__ import annotations

import argparse

import pytest
from fastapi.testclient import TestClient

from intergrax.scaffold.new_application import _PROFILES, create_application, register_parser
from tests.unit.applications.scaffold_runtime_helper import (
    factory_callable,
    import_scaffold_modules,
    lab_settings_class,
    prepare_scaffold_package,
    product_settings_class,
)

pytestmark = [pytest.mark.unit, pytest.mark.agent_os, pytest.mark.gate]


def _assert_tool_wiring_in_host(target, pkg: str, short: str) -> None:
    wiring = (target / "host" / "wiring.py").read_text(encoding="utf-8")
    tool_wiring = (target / "host" / "tool_wiring.py").read_text(encoding="utf-8")
    assert f"wire_{short}_tools" in wiring
    assert "tool_profile=tool_wiring.profile" in wiring
    assert "tool_wiring_context=tool_wiring.wiring_context" in wiring
    assert "ApplicationBuildContext.for_manifest" in wiring
    assert "build_application_tool_wiring" in tool_wiring
    assert "ToolProfile" in tool_wiring


def test_scaffold_profiles_exposed_on_cli() -> None:
    assert "lab" in _PROFILES
    assert "product" in _PROFILES
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register_parser(sub)
    app_parser = sub.choices["new-application"]
    profile_action = next(a for a in app_parser._actions if a.dest == "profile")
    assert set(profile_action.choices) == {"lab", "product"}


def test_scaffold_lab_profile_runtime_e2e(tmp_path) -> None:
    target, pkg, short = prepare_scaffold_package(
        tmp_path,
        name="gate_n9_lab",
        profile="lab",
        port=8191,
        route_prefix="/v1/gate_n9_lab",
    )
    factory_mod, settings_mod = import_scaffold_modules(pkg)
    settings_cls = lab_settings_class(settings_mod, short)
    settings = settings_cls(
        route_prefix="/v1/gate_n9_lab",
        include_mcp=False,
        include_scheduler=False,
        include_interaction_routes=False,
    )
    create_app = getattr(factory_mod, f"create_{short}_application")
    client = TestClient(create_app(settings=settings))

    prefix = "/v1/gate_n9_lab"
    agents_resp = client.get(f"{prefix}/agents")
    assert agents_resp.status_code == 200
    assert agents_resp.json()["agents"]

    run_resp = client.post(
        f"{prefix}/run",
        json={"message": "hello", "capability": "echo.basic"},
    )
    assert run_resp.status_code == 200
    assert run_resp.json().get("state") == "completed"

    sh = (target / "docker" / "build-docker.sh").read_text(encoding="utf-8")
    assert f'PKG="{pkg}"' in sh
    assert "applications/${PKG}/docker/Dockerfile" in sh
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert "build-docker.sh" in readme
    _assert_tool_wiring_in_host(target, pkg, short)


def test_scaffold_product_profile_runtime_e2e(tmp_path) -> None:
    target, pkg, short = prepare_scaffold_package(
        tmp_path,
        name="gate_n9_product",
        profile="product",
        port=8192,
        route_prefix="/v1/gate_n9_product",
    )
    factory_mod, settings_mod = import_scaffold_modules(pkg)
    from intergrax.fastapi_core.config import ApiEnvironment

    settings_cls = product_settings_class(settings_mod, short)
    settings = settings_cls(
        environment=ApiEnvironment.DEV,
        route_prefix="/v1/gate_n9_product",
        include_mcp=False,
    )
    create_app = factory_callable(factory_mod, short, product=True)
    client = TestClient(create_app(settings=settings))

    assert client.get("/health").status_code == 200

    prefix = "/v1/gate_n9_product"
    agents_resp = client.get(f"{prefix}/agents")
    assert agents_resp.status_code == 200
    assert agents_resp.json()["agents"]

    run_resp = client.post(
        f"{prefix}/run",
        json={"message": "hello", "capability": "echo.basic"},
    )
    assert run_resp.status_code == 200
    assert run_resp.json().get("state") == "completed"

    bat = (target / "docker" / "build-docker.bat").read_text(encoding="utf-8")
    assert pkg in bat
    assert "../../.." in (target / "docker" / "build-docker.sh").read_text(encoding="utf-8")
    _assert_tool_wiring_in_host(target, pkg, short)


def test_scaffold_generated_smoke_tests_are_importable(tmp_path) -> None:
    """Generated ``<pkg>_tests/host/test_*_smoke.py`` matches factory entrypoints."""
    _, pkg, short = prepare_scaffold_package(
        tmp_path,
        name="gate_n9_smoke",
        profile="lab",
        port=8193,
        route_prefix="/v1/gate_n9_smoke",
    )
    smoke_path = (
        tmp_path
        / "repo"
        / "applications"
        / pkg
        / f"{pkg}_tests"
        / "host"
        / f"test_{short}_host_smoke.py"
    )
    smoke_src = smoke_path.read_text(encoding="utf-8")
    assert f"create_{short}_application" in smoke_src
    assert "echo.basic" in smoke_src

    _, pkg2, short2 = prepare_scaffold_package(
        tmp_path / "product",
        name="gate_n9_smoke_prod",
        profile="product",
        port=8194,
        route_prefix="/v1/gate_n9_smoke_prod",
    )
    smoke2 = (
        tmp_path
        / "product"
        / "repo"
        / "applications"
        / pkg2
        / f"{pkg2}_tests"
        / "host"
        / f"test_{short2}_host_smoke.py"
    )
    smoke2_text = smoke2.read_text(encoding="utf-8")
    assert f"create_{short2}_backend_app" in smoke2_text
    assert f"test_{short2}_backend_health" in smoke2_text


def test_new_stack_cli_creates_agent_and_application(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "applications").mkdir(parents=True)
    from intergrax.scaffold.new_stack import run_new_stack
    import argparse

    code = run_new_stack(
        argparse.Namespace(
            name="stack_demo",
            capabilities=[],
            profile="lab",
            port=None,
            route_prefix=None,
            agent_only=False,
            app_only=False,
            root=root,
            force=False,
        )
    )
    assert code == 0
    assert (root / "agents" / "stack_demo").is_dir()
    assert (root / "applications" / "stack_demo_application" / "manifest.py").is_file()
    _assert_tool_wiring_in_host(
        root / "applications" / "stack_demo_application",
        "stack_demo_application",
        "stack_demo",
    )


def test_scaffold_rejects_unknown_profile(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "applications").mkdir()
    with pytest.raises(ValueError, match="Unsupported profile"):
        create_application(
            name="bad_profile",
            agents=["echo"],
            profile="enterprise",
            root=root,
        )
