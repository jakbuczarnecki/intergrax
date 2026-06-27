#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""One-off: replace getattr(_bundle, name) lazy exports in integration __init__ modules."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMPORT_LINE = "from intergrax.utils.lazy_export import export_from_bundle\n"


def patch_init(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "return getattr(_bundle, name)" not in text:
        return False
    if "export_from_bundle" not in text:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for index, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = index
                break
        lines.insert(insert_at, IMPORT_LINE)
        text = "".join(lines)
    text = text.replace(
        "return getattr(_bundle, name)",
        "return export_from_bundle(_bundle, name, _LAZY_EXPORTS)",
    )
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    changed = 0
    for path in sorted(REPO.glob("intergrax/integrations/**/__init__.py")):
        if patch_init(path):
            changed += 1
            print(path.relative_to(REPO).as_posix())
    print(f"Patched {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
