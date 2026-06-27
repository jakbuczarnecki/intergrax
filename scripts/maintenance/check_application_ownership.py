#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — PRODUCT manifests declare operational ownership (APP-OPS-2)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "applications", REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.ownership_wiring import (  # noqa: E402
    check_manifest_operational_ownership,
)
from intergrax.applications._shared.product_manifest_registry import (  # noqa: E402
    iter_product_manifests,
)


def main() -> int:
    violations: list[str] = []
    for product_id, manifest in iter_product_manifests():
        violations.extend(check_manifest_operational_ownership(product_id, manifest))

    if violations:
        print("application ownership gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application ownership gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
