#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — ApplicationRecoveryContract on STRICT product hosts (APP-EVOL-5)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.product_manifest_registry import (  # noqa: E402
    iter_strict_product_manifests,
)
from intergrax.applications._shared.recovery_contract_wiring import (  # noqa: E402
    check_strict_product_recovery_contract,
)


def main() -> int:
    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        violations.extend(
            check_strict_product_recovery_contract(
                product_id,
                manifest,
                applications_root=APPLICATIONS_ROOT,
            ),
        )

    if violations:
        print("application recovery contract gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application recovery contract gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
