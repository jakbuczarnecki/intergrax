#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""INT-MAINT-01 — STABLE integration slugs must have catalog honesty (description + probe evidence)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from intergrax.integrations.contracts.base import IntegrationStatus
    from intergrax.integrations.registry.bootstrap import register_default_integrations
    from intergrax.integrations.registry.catalog import iter_entries

    register_default_integrations(preset="full")
    violations: list[str] = []

    for entry in iter_entries():
        if entry.status is not IntegrationStatus.STABLE:
            continue
        if not (entry.description or "").strip():
            violations.append(f"{entry.slug}: STABLE without description")

    if violations:
        print("integration maturity label audit failed:")
        for item in sorted(violations)[:40]:
            print(f"  - {item}")
        if len(violations) > 40:
            print(f"  ... and {len(violations) - 40} more")
        return 1
    print("integration maturity label audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
