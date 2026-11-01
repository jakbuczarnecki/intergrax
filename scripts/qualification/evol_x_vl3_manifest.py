#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""W2-09: write/check ``evol_vl3_execution_manifest.v1.json`` (T3) — shared by CI and local runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testing_support.change_safety.selection.ci_manifest import (
    VL3_MANIFEST_FILE,
    build_manifest_document,
    collect_evol_x_node_ids,
    load_pinned_manifest,
    verify_manifest_integrity,
)

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "tests" / "qualification" / "evol_x" / "data" / VL3_MANIFEST_FILE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the pinned manifest from collect-only")
    mode.add_argument("--check", action="store_true", help="fail (exit 2) if pinned manifest != collect-only")
    args = parser.parse_args(argv)
    collected = collect_evol_x_node_ids()
    if args.write:
        doc = build_manifest_document(collected)
        _MANIFEST.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(f"wrote {len(collected)} node ids\n")
        return 0
    problems = verify_manifest_integrity(load_pinned_manifest(), collected)
    if problems:
        sys.stderr.write("VL3 manifest integrity FAILED: " + "; ".join(problems) + "\n")
        return 2
    sys.stdout.write(f"VL3 manifest OK ({len(collected)} node ids)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
