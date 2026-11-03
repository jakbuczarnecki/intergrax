#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Emit CHANGED-SINCE impact map SSOT (C024/C029)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.changed_since import (
    git_changed_files,
    invariant_floor,
    select_checks_for_changed_since,
    unmapped_evol_changed_paths,
)
from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = _repo_root()
    checks = load_default_manifest()
    try:
        selected = select_checks_for_changed_since(checks, repo_root=root, since_sha=args.since_sha)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    changed = git_changed_files(root, args.since_sha)
    unmapped = unmapped_evol_changed_paths(checks, changed)
    payload = {
        "schema_version": 1,
        "since_sha": args.since_sha,
        "selected_check_ids": sorted(selected),
        "invariant_floor_ids": sorted(invariant_floor(checks)),
        "unmapped_paths": sorted(unmapped),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
