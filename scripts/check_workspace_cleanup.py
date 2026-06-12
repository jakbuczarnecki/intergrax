#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — Tier-3 host factories wire workspace cleanup lifespan (APP-PROD-8)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.workspace_cleanup_wiring import (  # noqa: E402
    check_all_factory_workspace_cleanup,
)


def main() -> int:
    violations = check_all_factory_workspace_cleanup(APPLICATIONS_ROOT)
    if violations:
        print("workspace cleanup gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("workspace cleanup gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
