#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 security wiring from environment profile (Phase SEC-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.security_assembly_resolver import (
    assert_security_assembly_valid,
)
from intergrax.applications._shared.security_wiring import wire_application_security
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
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

    wiring = wire_application_security(env)
    assert_security_assembly_valid(wiring, env)

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
    assert_security_assembly_valid(wiring, env, nexus=runtime.nexus_loop)

    pipeline = runtime.nexus_loop._middleware  # noqa: SLF001
    if not isinstance(pipeline, MiddlewarePipeline):
        print("lab host NexusLoop must expose MiddlewarePipeline")
        return 1

    attached = {middleware.name for middleware in pipeline._middleware}  # noqa: SLF001
    for name in wiring.enabled_middleware:
        if name not in attached:
            print(f"lab host missing security middleware: {name}")
            return 1

    print("harness security wiring audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
