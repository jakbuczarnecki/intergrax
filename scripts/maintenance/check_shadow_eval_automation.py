#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-25.1 — shadow eval path automation gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    profile = ApplicationEnvironmentProfile.product_defaults(profile_id="gate.shadow")
    profile = profile.model_copy(
        update={
            "evaluation_profile": profile.evaluation_profile.model_copy(
                update={"shadow_eval_enabled": True, "online_registry_enabled": True}
            )
        }
    )
    wiring = wire_application_evaluation(profile)
    if not profile.evaluation_profile.shadow_eval_enabled:
        print("shadow_eval_enabled must be settable", file=sys.stderr)
        return 1
    if wiring.governance_bridge is None:
        print("governance_bridge required for shadow eval", file=sys.stderr)
        return 1
    trend_script = ROOT / "scripts" / "export_harness_shadow_eval_trend.py"
    if not trend_script.is_file():
        print("missing export_harness_shadow_eval_trend.py", file=sys.stderr)
        return 1
    print("OK: shadow eval automation wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
