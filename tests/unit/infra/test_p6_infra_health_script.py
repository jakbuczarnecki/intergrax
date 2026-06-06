# © Artur Czarnecki. All rights reserved.

"""Unit tests for optional P6 infra health script."""

from __future__ import annotations

import pytest

from scripts.check_p6_infra_health import ServiceProbeResult, collect_p6_infra_health

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_collect_p6_infra_health_returns_three_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.check_p6_infra_health._probe_http",
        lambda service, url: ServiceProbeResult(service=service, url=url, healthy=True, detail="HTTP 200"),
    )
    results = collect_p6_infra_health()
    assert len(results) == 3
    assert {item.service for item in results} == {"keycloak", "typesense", "airflow"}
