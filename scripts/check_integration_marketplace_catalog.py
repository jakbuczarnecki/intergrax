#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-13.1 — integration marketplace catalog + trust scoring."""

from __future__ import annotations

import sys

from intergrax.applications._shared.integration_marketplace_wiring import (
    resolve_integration_marketplace_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_integration_marketplace_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable integration marketplace catalog", file=sys.stderr)
        return 1
    catalog = wiring.catalog
    if catalog is None or not catalog.entries:
        print("integration marketplace catalog must include trust-scored entries", file=sys.stderr)
        return 1
    if any(entry.trust_score < 0.0 or entry.trust_score > 1.0 for entry in catalog.entries):
        print("integration trust scores must be normalized", file=sys.stderr)
        return 1
    print(f"OK: integration marketplace catalog ({len(catalog.entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
