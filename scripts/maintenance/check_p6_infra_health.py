#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Optional P6 Docker stack health probe (Keycloak, Typesense, Airflow)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]

P6_SERVICE_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("keycloak", "http://127.0.0.1:8088/health/ready"),
    ("typesense", "http://127.0.0.1:8108/health"),
    ("airflow", "http://127.0.0.1:8086/health"),
)


@dataclass(frozen=True, slots=True)
class ServiceProbeResult:
    service: str
    url: str
    healthy: bool
    detail: str


def _probe_http(service: str, url: str, *, timeout_seconds: float = 5.0) -> ServiceProbeResult:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            healthy = 200 <= status < 400
            return ServiceProbeResult(service=service, url=url, healthy=healthy, detail=f"HTTP {status}")
    except URLError as exc:
        return ServiceProbeResult(service=service, url=url, healthy=False, detail=str(exc.reason))


def collect_p6_infra_health() -> list[ServiceProbeResult]:
    return [_probe_http(service, url) for service, url in P6_SERVICE_ENDPOINTS]


def main() -> int:
    enabled = os.getenv("INTERGRAX_P6_INFRA_E2E", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        print(
            "p6 infra health: skipped (set INTERGRAX_P6_INFRA_E2E=true after "
            "./manage.sh start p6)"
        )
        return 0

    results = collect_p6_infra_health()
    failures: list[str] = []
    for item in results:
        status = "OK" if item.healthy else "FAIL"
        print(f"{status} {item.service}: {item.detail} ({item.url})")
        if not item.healthy:
            failures.append(item.service)

    if failures:
        print(f"p6 infra health: FAIL — unhealthy services: {', '.join(failures)}")
        return 1

    print("p6 infra health: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
