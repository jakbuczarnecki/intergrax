#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 critic wiring from environment profile (Phase CRIT-V-6.2 / CRIT-V-7.1)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.critic_assembly_resolver import assert_critic_assembly_valid
from intergrax.applications._shared.critic_wiring import wire_application_critic
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

    wiring = wire_application_critic(env)
    assert_critic_assembly_valid(wiring, env)

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

    critic_fragment = runtime.env_wiring.policy_bundle.domain_fragments.get("critic_governance")
    if not isinstance(critic_fragment, dict):
        print("lab host policy bundle must include critic_governance domain fragment")
        return 1

    if env.critic_profile.scopes.graph_final and runtime.critic.graph_hooks is None:
        print("lab host must wire critic graph hooks when graph_final scope enabled")
        return 1

    print("harness critic wiring audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
