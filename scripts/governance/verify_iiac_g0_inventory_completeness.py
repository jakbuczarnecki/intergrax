#!/usr/bin/env python3
"""Verify IIAC G0 matrix completeness (independent of generator discovery)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.governance.iiac_g0_inventory import (
    GOVERNANCE_IMPLEMENTATION_SHA,
    MANDATORY_IIAC_GOVERNANCE_PATHS,
    NORMATIVE_CATEGORY_REQUIREMENTS,
    REPO,
    enumerate_governed_paths_at_ref,
    git_merge_base_is_ancestor,
    git_rev_parse,
    unique_paths_in_entries,
)


def _load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_matrix_dict(
    manifest: dict,
    *,
    expect_head: str | None = None,
    allow_stale_source: bool = False,
) -> list[str]:
    errors: list[str] = []
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return ["matrix missing entries array"]

    source_sha = str(manifest.get("inventory_source_sha") or manifest.get("repository_head_at_generation") or "")
    if len(source_sha) != 40:
        errors.append("inventory_source_sha missing or not 40-char SHA")

    if expect_head and source_sha and not allow_stale_source:
        try:
            head = git_rev_parse(expect_head)
            if source_sha == head:
                pass
            elif not git_merge_base_is_ancestor(source_sha, head):
                errors.append(f"inventory_source_sha {source_sha[:12]}… not ancestor of HEAD {head[:12]}…")
        except RuntimeError as exc:
            errors.append(str(exc))

    if source_sha:
        try:
            expected_paths = set(enumerate_governed_paths_at_ref(source_sha))
        except RuntimeError as exc:
            errors.append(str(exc))
            expected_paths = set()
    else:
        expected_paths = set()

    covered = unique_paths_in_entries(entries)
    missing = sorted(expected_paths - covered)
    if missing:
        errors.append(f"missing {len(missing)} governed paths in matrix (first: {missing[0]})")

    extra = sorted(covered - expected_paths)
    if extra:
        errors.append(f"{len(extra)} matrix paths not in governed inventory (first: {extra[0]})")

    seen: set[tuple[int, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("invalid entry (not object)")
            continue
        cat = entry.get("category")
        path = entry.get("path")
        if not isinstance(cat, int) or cat not in range(1, 8):
            errors.append(f"invalid category for {path!r}: {cat!r}")
            continue
        if not isinstance(path, str) or not path.strip():
            errors.append("entry with empty path")
            continue
        key = (cat, path)
        if key in seen:
            errors.append(f"duplicate entry category={cat} path={path}")
        seen.add(key)

    for mandatory in sorted(MANDATORY_IIAC_GOVERNANCE_PATHS):
        if mandatory not in covered:
            errors.append(f"mandatory IIAC governance path missing: {mandatory}")

    for path, allowed_cats in NORMATIVE_CATEGORY_REQUIREMENTS.items():
        cats = {int(e["category"]) for e in entries if e.get("path") == path}
        if path in covered and not (cats & allowed_cats):
            errors.append(
                f"normative category mismatch for {path}: got {sorted(cats)}, need one of {sorted(allowed_cats)}"
            )

    gov_hist = manifest.get("governance_implementation_sha_historical")
    if gov_hist and str(gov_hist) != GOVERNANCE_IMPLEMENTATION_SHA:
        errors.append("governance_implementation_sha_historical must preserve Session A audit pin")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "matrix",
        type=Path,
        nargs="?",
        default=REPO / "docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json",
    )
    parser.add_argument("--expect-head", default="HEAD", help="Require inventory_source_sha == this ref")
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="Do not require inventory_source_sha == expect-head (fixtures only)",
    )
    args = parser.parse_args()

    if not args.matrix.is_file():
        print(f"matrix not found: {args.matrix}", file=sys.stderr)
        return 1

    manifest = _load_matrix(args.matrix)
    errors = validate_matrix_dict(
        manifest,
        expect_head=args.expect_head,
        allow_stale_source=args.allow_stale_source,
    )
    if errors:
        print("verify_iiac_g0_inventory_completeness: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    covered = len(unique_paths_in_entries(manifest.get("entries") or []))
    print(f"verify_iiac_g0_inventory_completeness: PASS ({covered} paths @ {manifest.get('inventory_source_sha')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
