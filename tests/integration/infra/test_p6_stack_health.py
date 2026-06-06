# © Artur Czarnecki. All rights reserved.

"""Integration tests for P6 Docker stack health (optional, requires running infra)."""

from __future__ import annotations

import os

import pytest

from scripts.check_p6_infra_health import collect_p6_infra_health

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("INTERGRAX_P6_INFRA_E2E", "").strip().lower() not in {"1", "true", "yes"},
    reason="Set INTERGRAX_P6_INFRA_E2E=true after ./manage.sh start p6",
)
def test_p6_docker_stack_services_healthy() -> None:
    results = collect_p6_infra_health()
    assert len(results) == 3
    unhealthy = [item.service for item in results if not item.healthy]
    assert not unhealthy, f"Unhealthy P6 services: {unhealthy}"
