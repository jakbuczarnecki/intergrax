#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 reliability wiring from environment profile (Phase REL-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.reliability_assembly_resolver import (
    assert_reliability_assembly_valid,
)
from intergrax.applications._shared.reliability_wiring import wire_application_reliability
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.contracts.idempotency_store import IdempotencyStore
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.wiring import bootstrap_lab_integration_wiring
from lab_application.manifest import build_lab_manifest


def main() -> int:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    if env is None:
        print("lab manifest missing environment profile")
        return 1

    wiring = wire_application_reliability(env)
    assert_reliability_assembly_valid(wiring, env)

    integrations = bootstrap_lab_integration_wiring(
        settings=settings,
        harness=settings.harness,
        otel_enabled=settings.otel_enabled,
    )
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        trace_db_path=integrations.trace_db_path,
        runtime_events_db_path=integrations.runtime_events_db_path,
        checkpoint_store=integrations.checkpoint_store,
        notification_adapter=integrations.notification_adapter,
    )

    if env.reliability_profile.idempotency_enabled:
        if not isinstance(runtime.reliability.idempotency_store, IdempotencyStore):
            print("lab host must wire idempotency store when idempotency_enabled")
            return 1

    if env.reliability_profile.long_running_scheduler_enabled:
        if resolve_harness_host_nexus_loop_legacy(runtime)._checkpoint_store is None:  # noqa: SLF001
            print("lab host must wire checkpoint store when long_running_scheduler_enabled")
            return 1

    print("harness reliability wiring audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
