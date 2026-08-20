# © Artur Czarnecki. All rights reserved.

"""Docker-backed vendor persistence proof for COMM-5 F3-E."""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

from intergrax.integrations.providers.security_status.client import HttpxSecurityStatusReadClient
from intergrax.integrations.providers.security_status.config import SecurityStatusIntegrationConfig
from intergrax.integrations.providers.security_status.integration import SecurityStatusIntegration
from intergrax.runtime.vendor_knowledge.live.errors import LiveErrorCodeV1
from intergrax.runtime.vendor_knowledge.live.failures import (
    LiveCallFailureReasonV1,
    live_call_failure_reason_for_error_code,
)
from proof_infrastructure.controlled_project_status_service.seed import ORION_FIXTURE_PROJECT_ID
from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorV1,
)


def _vendor_base_url() -> str:
    return os.environ.get(
        "GOVERNED_PROOF_SECURITY_VENDOR_URL",
        "http://127.0.0.1:8091",
    ).rstrip("/")


def _wait_for_vendor(timeout_seconds: float = 60.0) -> None:
    import time

    deadline = time.monotonic() + timeout_seconds
    last_error = "startup_timeout"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{_vendor_base_url()}/health", timeout=2.0)
        except httpx.HTTPError as exc:
            last_error = str(exc)
            continue
        if response.status_code == 200:
            return
        last_error = f"status={response.status_code}"
    raise RuntimeError(f"security_vendor_unavailable: {last_error}")


async def _read_snapshot_status() -> str:
    config = SecurityStatusIntegrationConfig(
        base_url=_vendor_base_url(),
        timeout_seconds=5.0,
    )
    client = HttpxSecurityStatusReadClient(config=config)
    integration = SecurityStatusIntegration.from_client(client, config=config)
    try:
        snapshot = await integration.read_security_status(
            project_id=ORION_FIXTURE_PROJECT_ID,
        )
    finally:
        await client.aclose()
    return snapshot.status


async def _run_scenarios() -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    def record(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        lines.append(f"{status} {name}: {detail}")

    _wait_for_vendor()
    record("docker_vendor_reachable", True, _vendor_base_url())

    response = httpx.get(
        f"{_vendor_base_url()}/projects/{ORION_FIXTURE_PROJECT_ID}/security-status",
        timeout=5.0,
    )
    record(
        "vendor_seed_record_readable",
        response.status_code == 200,
        f"status_code={response.status_code}",
    )
    seeded_status = response.json().get("status") if response.status_code == 200 else None

    integrated_status = await _read_snapshot_status()
    record(
        "intergrax_integration_reads_vendor",
        integrated_status == seeded_status,
        f"integrated={integrated_status}",
    )

    httpx.put(
        f"{_vendor_base_url()}/control/read-behavior",
        json={"behavior": SecurityStatusReadBehaviorV1.HTTP_503.value},
        timeout=5.0,
    )
    failure_reason = live_call_failure_reason_for_error_code(
        LiveErrorCodeV1.PROVIDER_TEMPORARILY_UNAVAILABLE.value,
    )
    record(
        "provider_failure_reason",
        failure_reason is LiveCallFailureReasonV1.PROVIDER_FAILED,
        failure_reason.value,
    )

    httpx.put(
        f"{_vendor_base_url()}/control/read-behavior",
        json={"behavior": SecurityStatusReadBehaviorV1.NORMAL.value},
        timeout=5.0,
    )
    restored = httpx.get(
        f"{_vendor_base_url()}/projects/{ORION_FIXTURE_PROJECT_ID}/security-status",
        timeout=5.0,
    )
    restored_status = restored.json().get("status") if restored.status_code == 200 else None
    record(
        "vendor_record_persisted_after_failure_recovery",
        restored.status_code == 200 and restored_status == seeded_status,
        f"status_code={restored.status_code}",
    )

    return ok, lines


def main() -> int:
    try:
        ok, lines = asyncio.run(_run_scenarios())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL docker_persistence_proof: {exc}")
        return 1
    for line in lines:
        print(line)
    print("PASS docker_persistence_proof" if ok else "FAIL docker_persistence_proof")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
