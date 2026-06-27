#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — Tier-3 ApplicationMigration schema and manifest coverage (APP-EVOL-2 · 2b)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.migration_wiring import (  # noqa: E402
    check_application_migrations,
)


def main() -> int:
    violations = check_application_migrations(APPLICATIONS_ROOT)
    if violations:
        print("application migrations gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application migrations gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
