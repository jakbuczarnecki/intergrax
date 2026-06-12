#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate: hosts with CE profile must resolve context engine (CE-12.4)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from intergrax.applications._shared.context_wiring import resolve_context_engine_from_environment
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.context.bootstrap import bootstrap_context_catalog, reset_context_catalog_bootstrap_for_tests

    reset_context_catalog_bootstrap_for_tests()
    bootstrap_context_catalog(register_shipped=True, discover_entry_points=False)
    errors: list[str] = []
    for preset in ("default", "codebase", "regulated_minimal", "explore_child"):
        env = ApplicationEnvironmentProfile.lab_defaults(profile_id=f"ce.gate.{preset}")
        env.context_profile.engine_preset = preset  # type: ignore[assignment]
        try:
            engine = resolve_context_engine_from_environment(env)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{preset}: {exc}")
            continue
        if engine.engine_id != preset:
            errors.append(f"{preset}: engine_id mismatch {engine.engine_id!r}")
    if errors:
        for err in errors:
            print(f"check_context_engine_wiring: FAIL — {err}", file=sys.stderr)
        return 1
    print("check_context_engine_wiring: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
