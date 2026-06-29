#!/usr/bin/env python3
"""Add contract factory to bundle lazy export sets in complex __init__.py files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from intergrax.integrations.providers.layout import SLUG_CATEGORY  # noqa: E402
from scripts.maintenance.migrate_provider_contract_integrations import contract_factory_name  # noqa: E402

PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"


def patch_init(path: Path, slug: str, category: str) -> bool:
    contract_factory = contract_factory_name(slug, category)
    text = path.read_text(encoding="utf-8")
    if contract_factory not in text:
        return False
    if f'"{contract_factory}"' in text and (
        f'"{contract_factory}",\n    }}' in text.replace(" ", "")
        or f'"{contract_factory}",' in text.split("_BUNDLE_EXPORTS")[1].split(")")[0]
        if "_BUNDLE_EXPORTS" in text
        else False
    ):
        # already in bundle exports block
        pass
    updated = text
    for set_name in ("_BUNDLE_EXPORTS", "_LAZY_EXPORTS"):
        if set_name not in updated:
            continue
        block = updated.split(set_name, 1)[1].split(")", 1)[0]
        if contract_factory in block:
            return False
        updated = re.sub(
            rf"({set_name} = frozenset\(\s*\{{)([^}}]*)(\}})",
            lambda m, s=contract_factory: m.group(1) + m.group(2).rstrip() + f'\n        "{s}",\n    ' + m.group(3),
            updated,
            count=1,
        )
        break
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    print(f"patched {path.relative_to(ROOT)}")
    return True


def main() -> None:
    count = 0
    for slug, category in SLUG_CATEGORY.items():
        if category == "observability_backend":
            continue
        init_path = PROVIDERS / category / slug / "__init__.py"
        if init_path.is_file() and patch_init(init_path, slug, category):
            count += 1
    print(f"patched {count} init files")


if __name__ == "__main__":
    main()
