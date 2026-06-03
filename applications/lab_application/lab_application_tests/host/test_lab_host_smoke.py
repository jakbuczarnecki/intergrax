# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from intergrax.fastapi_core.config import ApiEnvironment
from lab_application.host.factory import create_lab_application
from lab_application.host.settings import LabApplicationSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_application_lists_agents() -> None:
    client = TestClient(
        create_lab_application(settings=LabApplicationSettings(environment=ApiEnvironment.DEV))
    )
    response = client.get("/v1/lab/agents")
    assert response.status_code == 200
    assert response.json()["agents"]
