# © Artur Czarnecki. All rights reserved.

"""Runtime MCP mount checks — excluded from CI gate purity (TestClient).

MCP-RC MRC-5 (R-P7-02): built on the lab application (see ``test_mcp_surface_opt_in.py`` for why the strict
PRODUCT local-workspace app is no longer a usable vehicle for an MCP opt-in check).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@pytest.fixture(autouse=True)
def _explicit_provider_and_optional_sdk_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("ollama") is None:
        stub = types.ModuleType("ollama")
        stub.Client = lambda *_args, **_kwargs: object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ollama", stub)
    monkeypatch.setenv("INTERGRAX_LLM_PROVIDER", "ollama")


def _lab_app(*, include_mcp: bool):
    from lab_application.host.factory import create_lab_application
    from lab_application.host.settings import LabApplicationSettings

    settings = dataclasses.replace(LabApplicationSettings.from_env(), include_mcp=include_mcp)
    return create_lab_application(settings=settings)


def _route_paths(app) -> set[str]:  # type: ignore[no-untyped-def]
    return {path for route in app.routes if isinstance(path := getattr(route, "path", None), str)}


def test_mcp_enabled_factory_mounts_mcp_route_when_available() -> None:
    app = _lab_app(include_mcp=True)
    # the mounted ASGI app answers: an unauthenticated GET on the MCP endpoint is a protocol-level response, not a 404
    response = TestClient(app, raise_server_exceptions=False).get("/mcp/")
    assert response.status_code != 404
    assert {"/mcp", "/mcp/"} & _route_paths(app)


def test_mcp_disabled_factory_does_not_mount_mcp_route() -> None:
    app = _lab_app(include_mcp=False)
    assert not {"/mcp", "/mcp/"} & _route_paths(app)
