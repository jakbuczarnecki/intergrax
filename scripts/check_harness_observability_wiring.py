#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 observability wiring from environment profile (Phase OBS-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.observability_assembly_resolver import (
    assert_observability_assembly_valid,
)
from intergrax.applications._shared.observability_wiring import wire_application_observability
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest


def main() -> int:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    if env is None:
        print("lab manifest missing environment profile")
        return 1

    wiring = wire_application_observability(env)
    assert_observability_assembly_valid(wiring, env)

    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    if not isinstance(runtime.observability.trace_store, (SQLiteRunTraceStore,)):
        if env.observability_profile.trace_sqlite_enabled:
            print("lab host must use SQLite trace store when trace_sqlite_enabled")
            return 1

    if runtime.observability.runtime_event_store is None:
        if env.observability_profile.trace_sqlite_enabled:
            print("lab host must wire runtime event journal when trace_sqlite_enabled")
            return 1

    print("harness observability wiring audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
