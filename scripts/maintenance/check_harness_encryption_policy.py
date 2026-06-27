#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Validate encryption enforcement assembly on strict production security profile (Phase ENC-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.applications._shared.security_assembly_resolver import (
    SecurityAssemblyError,
    assert_security_assembly_valid,
)
from intergrax.applications._shared.security_wiring import wire_application_security
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import SecurityEnvelope
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.presets import harness_defense_stack


def main() -> int:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.encryption.smoke")
    env.meta = env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env.security = SecurityEnvelope.production()
    env.integration_profile = harness_defense_stack()

    wiring = wire_application_security(env)
    try:
        assert_security_assembly_valid(wiring, env)
    except SecurityAssemblyError as exc:
        print(f"encryption policy assembly failed: {exc}")
        return 1

    env_no_secrets = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.encryption.fail")
    env_no_secrets.meta = env_no_secrets.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env_no_secrets.security = SecurityEnvelope.production()
    wiring_fail = wire_application_security(env_no_secrets)
    try:
        assert_security_assembly_valid(wiring_fail, env_no_secrets)
    except SecurityAssemblyError:
        print("harness encryption policy audit: OK")
        return 0
    print("expected strict host without secrets_store to fail assembly")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
