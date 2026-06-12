#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — STRICT product hosts pass capability graph deploy review (APP-OPS-1)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "applications", REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.capability_graph_deploy_gate import (  # noqa: E402
    check_strict_product_capability_graph,
)
from intergrax.applications._shared.product_manifest_registry import (  # noqa: E402
    iter_strict_product_manifests,
)


def main() -> int:
    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        violations.extend(check_strict_product_capability_graph(product_id, manifest))

    if violations:
        print("capability graph strict deploy gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("capability graph strict deploy gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
