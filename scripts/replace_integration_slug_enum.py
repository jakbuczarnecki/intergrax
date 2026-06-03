#!/usr/bin/env python3
"""Remove IntegrationSlug enum references — use string slugs or manifests."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}


def _slug_from_member(member: str) -> str:
    return member.lower()


def _transform(text: str) -> tuple[str, bool]:
    original = text
    text = re.sub(
        r"IntegrationSlug\.([A-Z0-9_]+)\.value",
        lambda m: f'"{_slug_from_member(m.group(1))}"',
        text,
    )
    text = re.sub(
        r"IntegrationSlug\.([A-Z0-9_]+)",
        lambda m: f'"{_slug_from_member(m.group(1))}"',
        text,
    )
    # Drop dedicated enum imports
    text = re.sub(
        r"from intergrax\.integrations\.registry\.slugs import IntegrationSlug\n",
        "",
        text,
    )
    text = re.sub(
        r"from intergrax\.integrations\.registry\.slugs import ([^;\n]+)\n",
        lambda m: _fix_import_line(m.group(1)),
        text,
    )
    text = re.sub(
        r",\s*IntegrationSlug",
        "",
        text,
    )
    text = re.sub(
        r"IntegrationSlug,\s*",
        "",
        text,
    )
    return text, text != original


def _fix_import_line(imports: str) -> str:
    parts = [p.strip() for p in imports.split(",")]
    kept = [p for p in parts if p and p != "IntegrationSlug"]
    if not kept:
        return ""
    return f"from intergrax.integrations.registry.slugs import {', '.join(kept)}\n"


def main() -> int:
    changed = 0
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if "replace_integration_slug_enum" in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        if "IntegrationSlug" not in text:
            continue
        new_text, did = _transform(text)
        if did:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
            print(path.relative_to(ROOT))
    print(f"updated {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
