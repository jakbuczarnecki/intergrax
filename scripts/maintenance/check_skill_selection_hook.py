#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-12.2 — dynamic skill selection L4 hook."""

from __future__ import annotations

import sys

from intergrax.applications._shared.skill_selection_wiring import resolve_skill_selection_hook
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.adaptive.skill_selection_engine import SkillSelectionEngine


def main() -> int:
    hook = resolve_skill_selection_hook(ApplicationEnvironmentProfile.product_defaults())
    if not hook.enabled:
        print("product host must enable skill selection hook", file=sys.stderr)
        return 1
    if hook.engine_id != SkillSelectionEngine().engine_id:
        print("skill selection hook must reference skill_selection engine", file=sys.stderr)
        return 1
    if not hook.candidate_bundles:
        print("skill selection hook must expose candidate bundles", file=sys.stderr)
        return 1

    print("OK: skill selection L4 hook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
