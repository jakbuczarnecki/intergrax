#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""INT-MAINT-03 — SaaS-only slugs must not claim local container requirement."""

from __future__ import annotations

import sys


def main() -> int:
    from intergrax.integrations._shared.saas_only_slugs import SAAS_ONLY_SLUGS
    from intergrax.integrations.registry.bootstrap import register_default_integrations
    from intergrax.integrations.registry.catalog import get_entry

    register_default_integrations(preset="full")
    violations: list[str] = []

    for slug in sorted(SAAS_ONLY_SLUGS):
        try:
            entry = get_entry(slug)
        except KeyError:
            violations.append(f"{slug}: SaaS slug missing from catalog")
            continue
        if entry.requires_local_container:
            violations.append(f"{slug}: SaaS-only slug must not require local container")

    if violations:
        print("integration SaaS honesty audit failed:")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("integration SaaS honesty audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
