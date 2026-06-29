# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from research_application.host.factory import create_research_backend_app

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_research_application_health() -> None:
    client = TestClient(create_research_backend_app())
    response = client.get("/health")
    assert response.status_code == 200
