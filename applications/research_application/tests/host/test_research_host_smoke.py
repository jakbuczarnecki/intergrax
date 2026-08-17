# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from research_application.host.factory import create_research_backend_app
from research_application.tests.research_ac3_projection import build_research_test_registry_projection

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_research_application_health() -> None:
    client = TestClient(
        create_research_backend_app(
            registry_projection=build_research_test_registry_projection(),
        )
    )
    response = client.get("/health")
    assert response.status_code == 200
