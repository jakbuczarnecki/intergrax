#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-20.2 — capability edge catalog sync gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from intergrax.runtime.architecture.capability_edge_catalog import (
        build_edge_catalog,
        catalog_path,
    )

    path = catalog_path(REPO_ROOT)
    if not path.is_file():
        print(f"missing catalog: {path}", file=sys.stderr)
        return 1
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    current = build_edge_catalog()
    if on_disk != current:
        print("capability edge catalog out of sync; regenerate via capability_edge_catalog.write_catalog", file=sys.stderr)
        return 1
    print(f"OK: {len(current['edges'])} capability edges cataloged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
