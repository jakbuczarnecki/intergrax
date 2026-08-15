# © Artur Czarnecki. All rights reserved.
"""Validate harness ADR layout: entries/YYYY-MM-DD/*.md + README index links."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADR_ROOT = ROOT / "docs" / "project" / "technical" / "adr"
README = ADR_ROOT / "README.md"
ENTRIES = ADR_ROOT / "entries"

INDEX_LINK_RE = re.compile(
    r"\[(ADR-[A-Z0-9]+-\d+)\]\((entries/\d{4}-\d{2}-\d{2}/ADR-[A-Z0-9]+-\d+\.md)\)"
)
STRAY_ROOT_ADR_RE = re.compile(r"^ADR-[A-Z0-9]+-\d+\.md$")


def main() -> int:
    errors: list[str] = []

    if not README.is_file():
        errors.append("missing docs/project/technical/adr/README.md")
        _report(errors)
        return 1

    if not ENTRIES.is_dir():
        errors.append("missing docs/project/technical/adr/entries/")
        _report(errors)
        return 1

    indexed: set[str] = set()
    readme_text = README.read_text(encoding="utf-8")
    for _adr_id, rel_path in INDEX_LINK_RE.findall(readme_text):
        path = ADR_ROOT.joinpath(*rel_path.split("/"))
        if not path.is_file():
            errors.append(f"README index links missing file: {rel_path}")
        else:
            indexed.add(path.name)

    on_disk: set[str] = set()
    for day_dir in sorted(ENTRIES.iterdir()):
        if not day_dir.is_dir():
            errors.append(f"non-directory under entries/: {day_dir.name}")
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_dir.name):
            errors.append(f"invalid entries day folder: {day_dir.name}")
            continue
        for adr_file in sorted(day_dir.glob("ADR-*.md")):
            on_disk.add(adr_file.name)

    for basename in sorted(on_disk - indexed):
        errors.append(f"ADR on disk but missing from README index: {basename}")
    for basename in sorted(indexed - on_disk):
        errors.append(f"README index entry missing on disk: {basename}")

    for path in ADR_ROOT.iterdir():
        if path.is_file() and STRAY_ROOT_ADR_RE.match(path.name):
            errors.append(f"harness ADR must live under entries/: {path.name}")

    _report(errors)
    return 1 if errors else 0


def _report(errors: list[str]) -> None:
    if errors:
        print("Harness ADR check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
    else:
        print("Harness ADR check OK")


if __name__ == "__main__":
    raise SystemExit(main())
