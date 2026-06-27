#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — ApplicationRegistry and EnvironmentRegistry sync (APP-OPS-4)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "applications", REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.registry_ops_wiring import check_platform_registries  # noqa: E402


def main() -> int:
    violations = check_platform_registries(REPO_ROOT)

    if violations:
        print("application registry gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application registry gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
