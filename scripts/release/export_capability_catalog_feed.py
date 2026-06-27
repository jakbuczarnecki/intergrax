#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Export integration/tool/skill slug feed for UI builder (Phase DX-7.4)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.integrations.registry.catalog import list_slugs as list_integration_slugs
from intergrax.tools.registry.catalog import list_catalog_tool_ids


def main() -> int:
    bootstrap_catalogs(register_shipped=True, integration_preset="core")
    feed = {
        "integrations": [{"slug": s} for s in sorted(list_integration_slugs())],
        "tools": [{"tool_id": t} for t in sorted(list_catalog_tool_ids())],
        "skills": [],
    }
    out = Path(__file__).resolve().parents[2] / "build" / "harness_specs" / "capability_catalog_feed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(feed, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
