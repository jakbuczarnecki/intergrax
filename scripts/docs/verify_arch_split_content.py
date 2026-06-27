#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Verify architecture hub+satellite splits preserve canon body (F4)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from arch_hub_config import CONFIGS
from arch_hub_lib import ARCH_DIR, SAT_DIR, merge_arch_satellites

ROOT = Path(__file__).resolve().parents[2]

# Strip navigation-only blocks that differ between runs.
STRIP_PATTERNS = (
    re.compile(r"## Architecture satellites \(read on demand\).*?(?=\n## |\n# |\Z)", re.DOTALL),
    re.compile(r"^# .+ — .+\n\n\*\*Parent hub:\*\* \[`.+`\]\(\../.+\.md\)\n\n", re.MULTILINE),
)


def normalize(text: str) -> str:
    for pat in STRIP_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def merged_chars(domain: str) -> int:
    return len(normalize(merge_arch_satellites(domain)))


def main() -> int:
    errors: list[str] = []
    for domain in sorted(CONFIGS):
        hub = ARCH_DIR / f"{domain}.md"
        if not hub.exists():
            continue
        merged = merged_chars(domain)
        hub_only = len(normalize(hub.read_text(encoding="utf-8")))
        sat_only = 0
        for p in SAT_DIR.glob(f"{domain}_*.md"):
            sat_only += len(normalize(p.read_text(encoding="utf-8")))
        if merged < int((hub_only + sat_only) * 0.88):
            errors.append(
                f"{domain}: merged {merged} << hub+sat {hub_only + sat_only} (possible content loss)"
            )
        if not list(SAT_DIR.glob(f"{domain}_*.md")) and hub_only > 15_000:
            errors.append(f"{domain}: large hub ({hub_only} chars) with no satellites")
    if errors:
        print("verify_arch_split_content: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("verify_arch_split_content: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
