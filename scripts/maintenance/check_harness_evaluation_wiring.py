#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 evaluation wiring from environment profile (Phase EVAL-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.evaluation_assembly_resolver import assert_evaluation_assembly_valid
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
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

    wiring = wire_application_evaluation(env)
    assert_evaluation_assembly_valid(wiring, env)

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

    if env.evaluation_profile.online_registry_enabled and runtime.evaluation.registry is None:
        print("lab host must wire evaluation registry when online_registry_enabled")
        return 1

    if env.evaluation_profile.shadow_eval_enabled and runtime.evaluation.governance_bridge is None:
        print("lab host must wire governance bridge when shadow_eval_enabled")
        return 1

    eval_fragment = runtime.env_wiring.policy_bundle.domain_fragments.get("evaluation_governance")
    if not isinstance(eval_fragment, dict):
        print("lab host policy bundle must include evaluation_governance domain fragment")
        return 1

    print("harness evaluation wiring audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
