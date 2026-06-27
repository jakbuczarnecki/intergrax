#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-13.2 — catalog hot-reload without host restart."""

from __future__ import annotations

import sys

from intergrax.applications._shared.catalog_hot_reload_wiring import resolve_catalog_hot_reload_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_catalog_hot_reload_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable catalog hot-reload", file=sys.stderr)
        return 1
    report = wiring.report
    if report is None or not report.reloaded or report.after_count <= 0:
        print("catalog hot-reload report invalid", file=sys.stderr)
        return 1
    print(f"OK: catalog hot-reload ({report.after_count} slugs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
