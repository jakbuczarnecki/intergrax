#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Verify enterprise manifest check_ids align with Annex C control matrix."""

from __future__ import annotations

import json
import re
import sys
from importlib import resources
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _annex_c_ids(root: Path) -> set[str]:
    path = root / "docs/project/maintainers/qualification/ent_audit_x/ANNEX_C_CONTROL_MATRIX.md"
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 3:
            continue
        token = cells[1]
        if re.match(r"^(ENT-A|TEN-|VBN-)", token):
            ids.add(token)
    return ids


def main() -> int:
    root = _repo_root()
    pkg = resources.files("intergrax.platform.enterprise_audit")
    manifest = json.loads((pkg / "data" / "enterprise_audit_check_manifest.v1.json").read_text(encoding="utf-8"))
    manifest_ids = {row["check_id"] for row in manifest["checks"]}
    annex_ids = _annex_c_ids(root)
    if manifest_ids != annex_ids:
        missing = annex_ids - manifest_ids
        extra = manifest_ids - annex_ids
        print(f"manifest/annex mismatch missing={missing} extra={extra}", file=sys.stderr)
        return 1
    if len(manifest_ids) != 85:
        print(f"expected 85 checks, got {len(manifest_ids)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
