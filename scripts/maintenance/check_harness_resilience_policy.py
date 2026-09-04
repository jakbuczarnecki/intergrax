#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit resilience policy and autonomy wiring (REL-ADV.6)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.resilience_policy import ResiliencePolicy
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.task.task import Task, TaskContext
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

    policy = env.reliability_profile.resilience_policy
    if not isinstance(policy, ResiliencePolicy):
        print("reliability_profile.resilience_policy must be ResiliencePolicy")
        return 1

    middleware = resolve_harness_host_nexus_loop_legacy(runtime)._middleware  # noqa: SLF001
    if not isinstance(middleware, MiddlewarePipeline):
        print("expected MiddlewarePipeline on nexus loop")
        return 1
    names = {item.name for item in middleware._middleware}  # noqa: SLF001
    if "AutonomyGovernanceMiddleware" not in names:
        print("AutonomyGovernanceMiddleware must be registered on lab host")
        return 1

    task = apply_reliability_task_defaults(
        Task(
            task_id="audit_task",
            tenant_id="lab",
            user_id="audit",
            message="audit",
            context=TaskContext(capability="echo.basic"),
        ),
        env,
    )
    if task.options.governance.autonomy_level is None:
        print("apply_reliability_task_defaults must set autonomy_level")
        return 1
    if task.metadata.get("resilience_policy.v1") is None:
        print("apply_reliability_task_defaults must embed resilience_policy.v1")
        return 1

    async_env = ApplicationEnvironmentProfile.async_batch_defaults()
    if not async_env.orchestration_profile.long_running_enabled:
        print("async_batch_defaults must enable long_running")
        return 1

    if env.reliability_profile.default_autonomy_level not in AutonomyLevel:
        print("invalid default_autonomy_level on lab environment")
        return 1

    print("harness resilience policy audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
