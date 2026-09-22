#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-13.2 — governed catalog hot-reload capability wiring."""

from __future__ import annotations

import sys

from intergrax.applications._shared.catalog_hot_reload_wiring import resolve_catalog_hot_reload_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_catalog_hot_reload_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable catalog hot-reload", file=sys.stderr)
        return 1
    if wiring.service is None:
        print("catalog hot-reload governed service missing", file=sys.stderr)
        return 1
    print("OK: catalog hot-reload capability (governed service; operator invocation required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
