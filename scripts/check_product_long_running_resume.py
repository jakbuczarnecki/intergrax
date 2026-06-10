#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-8.1 — long-running workflow resume on product hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.product_long_running_wiring import resolve_product_long_running_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    wiring = resolve_product_long_running_wiring(env)
    if not wiring.scheduler_enabled:
        print("product host must enable long-running scheduler", file=sys.stderr)
        return 1
    if not wiring.checkpoint_resume_enabled:
        print("product host must enable checkpoint resume", file=sys.stderr)
        return 1
    print("OK: product long-running resume wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
