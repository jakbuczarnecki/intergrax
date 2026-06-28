#!/usr/bin/env python3
"""Restore correct _shared.pN.factories imports in migrated simple bundles."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"

IMPORT_RE = re.compile(
    r"from intergrax\.integrations\._shared\.p\d+\.factories import \w+",
)


def git_head(path: Path) -> str | None:
    rel = path.relative_to(ROOT).as_posix()
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            text=True,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError:
        return None


def main() -> None:
    fixed = 0
    for bundle in sorted(PROVIDERS.rglob("bundle.py")):
        current = bundle.read_text(encoding="utf-8")
        if "create_" not in current or "_integration" not in current:
            continue
        original = git_head(bundle)
        if original is None:
            continue
        orig_match = IMPORT_RE.search(original)
        if orig_match is None:
            continue
        if orig_match.group(0) in current:
            continue
        if not IMPORT_RE.search(current):
            continue
        updated = IMPORT_RE.sub(orig_match.group(0), current, count=1)
        if updated != current:
            bundle.write_text(updated, encoding="utf-8")
            fixed += 1
            print(f"fixed {bundle.relative_to(ROOT)}")
    print(f"restored {fixed} bundle imports")


if __name__ == "__main__":
    main()
