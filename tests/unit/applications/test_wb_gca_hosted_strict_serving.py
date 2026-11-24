# © Artur Czarnecki. All rights reserved.

"""W-B Part G — Package B strict production on the hosted runner (NT-WB-26)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingReadiness,
    StrictProductionServingSession,
)

from governed_contractor_application.host import package_b_strict_production_serving_command as pb_cmd
from governed_contractor_application.hosting.profile import build_governed_contractor_hosted_profile

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVE_COMMAND = (
    _REPO_ROOT / "applications/governed_contractor_application/host/package_b_strict_production_serving_command.py"
)


def _is_uvicorn_run(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "uvicorn"
    )


def _ready_session() -> StrictProductionServingSession:
    session = MagicMock(spec=StrictProductionServingSession)
    session.readiness = StrictProductionServingReadiness(
        process_lifecycle_open=True,
        production_dependencies_validated=True,
        serving_state_reconciled=True,
        serving_handoff_bound=True,
        execution_admission_eligible=True,
        ready_for_traffic=True,
    )
    session.is_closed = False
    return session


def test_nt_wb_26_package_b_serves_through_hosted_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOVERNED_CONTRACTOR_BACKEND_BIND_HOST", raising=False)
    monkeypatch.setattr(
        "governed_contractor_application.hosting.profile.compose_governed_contractor_host_runtime",
        lambda _settings: None,
    )
    captured_profiles: list[object] = []
    session = _ready_session()
    settings = SimpleNamespace(
        api_keys_map={"k": object()},
        backend_port=8000,
        strict_external_work_provider_id="pinned",
        strict_external_work_provider_descriptor_digest="digest",
    )

    def _fake_hosted(*, session: object, settings: object | None = None, **kwargs: object) -> SimpleNamespace:
        captured_profiles.append(
            build_governed_contractor_hosted_profile(session=session, settings=settings),  # type: ignore[arg-type]
        )
        return SimpleNamespace()

    monkeypatch.setattr(
        pb_cmd,
        "GovernedContractorBackendSettings",
        MagicMock(from_env=MagicMock(return_value=settings)),
    )
    monkeypatch.setattr(pb_cmd, "_validate_operator_backend_settings", lambda _s: None)
    monkeypatch.setattr(
        pb_cmd,
        "load_package_b_operator_production_control_plane_governance_from_env",
        lambda: object(),
    )
    monkeypatch.setattr(pb_cmd, "load_strict_production_serving_environment_from_env", lambda: object())
    monkeypatch.setattr(
        pb_cmd,
        "load_single_node_production_profile_v1_from_env",
        lambda: SimpleNamespace(trusted_artifact_parent_roots=()),
    )
    monkeypatch.setattr(pb_cmd, "validate_configured_single_node_production_profile_v1", lambda _p: None)
    monkeypatch.setattr(pb_cmd, "VenvBundleColocatedRuntimeDeploymentAdapter", MagicMock())
    monkeypatch.setattr(
        pb_cmd,
        "open_gca_strict_production_serving_session_from_single_node_production_profile_v1",
        lambda **_kwargs: (session, MagicMock(), MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(pb_cmd, "run_governed_contractor_hosted_strict_production_application", _fake_hosted)
    monkeypatch.setattr("uvicorn.run", lambda *_a, **_k: pytest.fail("direct uvicorn.run must not run"))

    pb_cmd.run_package_b_strict_production_serving_command()
    assert len(captured_profiles) == 1
    factory = getattr(captured_profiles[0], "application_factory")
    assert getattr(factory, "bind_host") == "127.0.0.1"


def test_nt_wb_26_command_has_no_direct_uvicorn_run() -> None:
    tree = ast.parse(_SERVE_COMMAND.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_package_b_strict_production_serving_command"
    )
    assert not any(_is_uvicorn_run(node) for node in ast.walk(function))
    assert "run_governed_contractor_hosted_strict_production_application" in ast.unparse(function)


def test_nt_wb_26_explicit_all_interfaces_bind_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOVERNED_CONTRACTOR_BACKEND_BIND_HOST", "0.0.0.0")
    monkeypatch.setattr(
        "governed_contractor_application.hosting.profile.compose_governed_contractor_host_runtime",
        lambda _settings: None,
    )
    settings = SimpleNamespace(backend_port=8000)
    session = _ready_session()
    profile = build_governed_contractor_hosted_profile(session=session, settings=settings)  # type: ignore[arg-type]
    assert getattr(profile.application_factory, "bind_host") == "0.0.0.0"
