#!/usr/bin/env python3

# © Artur Czarnecki. All rights reserved.



"""Smoke security defense plugin registry and lab strict profile (Phase SEC-EXT-6 / SEC-EVOL-6)."""



from __future__ import annotations



import sys

from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[2]

for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):

    path_text = str(path)

    if path_text not in sys.path:

        sys.path.insert(0, path_text)



from intergrax.applications._shared.security_assembly_resolver import assert_security_assembly_valid

from intergrax.applications._shared.security_wiring import wire_application_security

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

from intergrax.applications.contracts.environment_profile.bundles import SecurityEnvelope

from intergrax.core.catalog_bootstrap import bootstrap_catalogs

from intergrax.integrations.registry.bootstrap import register_default_integrations

from intergrax.integrations.registry.presets import harness_defense_stack

from intergrax.runtime.security.defense_registry import get_security_defense_plugin





def main() -> int:

    register_default_integrations()

    result = bootstrap_catalogs(discover_entry_points=False)

    if get_security_defense_plugin("harness.strict_injection") is None:

        print("harness.strict_injection plugin not resolvable")

        return 1

    if result.security_entry_point_plugins < 0:

        print("invalid security entry point count")

        return 1



    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.defense.smoke")

    env.security = SecurityEnvelope.lab()

    env.security = env.security.model_copy(

        update={"application_security": SecurityEnvelope.production().application_security},

    )

    env.integration_profile = harness_defense_stack()



    wiring = wire_application_security(env)

    assert_security_assembly_valid(wiring, env)

    print("harness security defense plugins audit: OK")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

