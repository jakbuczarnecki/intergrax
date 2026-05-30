# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_application_exposes_mcp_mount() -> None:
    from lab_application.host.factory import create_lab_application
    from lab_application.host.settings import LabApplicationSettings

    settings = LabApplicationSettings(include_mcp=True, include_scheduler=False)
    app = create_lab_application(settings=settings)
    client = TestClient(app)
    assert client.get("/v1/lab/agents").status_code == 200
    assert any(getattr(r, "path", None) in {"/mcp", "/mcp/"} for r in app.routes if hasattr(r, "path"))


def test_research_application_exposes_mcp_mount() -> None:
    from research_application.host.factory import create_research_backend_app
    from research_application.host.settings import ResearchBackendSettings

    settings = ResearchBackendSettings(include_mcp=True)
    app = create_research_backend_app(settings=settings)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert any(getattr(r, "path", None) in {"/mcp", "/mcp/"} for r in app.routes if hasattr(r, "path"))
