# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from intergrax.utils import attribute_access

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_lab_application_exposes_mcp_mount() -> None:
    from lab_application.host.factory import create_lab_application
    from lab_application.host.settings import LabApplicationSettings

    from intergrax.fastapi_core.config import ApiEnvironment

    settings = LabApplicationSettings(
        environment=ApiEnvironment.DEV,
        include_mcp=True,
        include_scheduler=False,
    )
    app = create_lab_application(settings=settings)
    client = TestClient(app)
    assert client.get("/v1/lab/agents").status_code == 200
    assert any(attribute_access.optional(r, "path", None) in {"/mcp", "/mcp/"} for r in app.routes if hasattr(r, "path"))


def test_research_application_exposes_mcp_mount(
    monkeypatch: pytest.MonkeyPatch,
    harness_auth_headers: dict[str, str],
) -> None:
    from research_application.host.factory import create_research_backend_app
    from research_application.host.settings import ResearchBackendSettings

    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "gate-test-harness-key")
    settings = ResearchBackendSettings(include_mcp=True)
    app = create_research_backend_app(settings=settings)
    client = TestClient(app, headers=harness_auth_headers)
    assert client.get("/health").status_code == 200
    assert any(attribute_access.optional(r, "path", None) in {"/mcp", "/mcp/"} for r in app.routes if hasattr(r, "path"))
