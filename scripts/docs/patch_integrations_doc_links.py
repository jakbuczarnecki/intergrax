#!/usr/bin/env python3
"""Patch docs/project/architecture/INTEGRATIONS.md with USAGE.md links for all providers (category layout)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from intergrax.integrations.providers.layout import SLUG_CATEGORY, provider_package_path

DOCS = ROOT / "docs" / "project" / "architecture/INTEGRATIONS.md"
PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"


def usage_link(slug: str) -> str:
    return f"[USAGE.md](../{provider_package_path(slug)}/USAGE.md)"


def usage_index_link(slug: str) -> str:
    return f"[USAGE](../{provider_package_path(slug)}/USAGE.md)"


def main() -> None:
    text = DOCS.read_text(encoding="utf-8")

    for slug in sorted(SLUG_CATEGORY.keys(), key=len, reverse=True):
        pattern = rf"(\| `{slug}` \| [^\n]+) \| — \|"
        repl = rf"\1 | {usage_index_link(slug)} |"
        text = re.sub(pattern, repl, text)

    DOCS.write_text(text, encoding="utf-8")
    missing = [
        slug
        for slug, category in SLUG_CATEGORY.items()
        if not (PROVIDERS / category / slug / "USAGE.md").exists()
    ]
    print(f"updated {DOCS.relative_to(ROOT)}")
    if missing:
        print("MISSING USAGE:", missing)
    else:
        print("all providers have USAGE.md")


if __name__ == "__main__":
    main()
