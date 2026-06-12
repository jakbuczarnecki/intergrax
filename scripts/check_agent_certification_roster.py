#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — STRICT product roster agent certification (APP-EVOL-4)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.agent_certification_wiring import (  # noqa: E402
    check_strict_product_agent_certification,
)
from intergrax.applications._shared.product_manifest_registry import (  # noqa: E402
    iter_strict_product_manifests,
)


def main() -> int:
    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        violations.extend(check_strict_product_agent_certification(product_id, manifest))

    if violations:
        print("agent certification roster gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("agent certification roster gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
