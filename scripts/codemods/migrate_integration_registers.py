#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Migrate provider register.py to co-located manifest.py + register_from_manifest."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"


def _find_register_files() -> list[Path]:
    files: list[Path] = []
    for path in PROVIDERS.rglob("register.py"):
        text = path.read_text(encoding="utf-8")
        if "register_from_manifest" in text and "IntegrationEntry" not in text:
            continue
        if "IntegrationEntry" not in text:
            continue
        files.append(path)
    return sorted(files)


def _extract_via_ast(text: str) -> dict[str, str] | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_entry = (isinstance(func, ast.Name) and func.id == "IntegrationEntry") or (
            isinstance(func, ast.Attribute) and func.attr == "IntegrationEntry"
        )
        if not is_entry:
            continue
        fields: dict[str, ast.expr] = {}
        for kw in node.keywords:
            fields[kw.arg or ""] = kw.value

        def _lit(key: str) -> str | None:
            expr = fields.get(key)
            if expr is None:
                return None
            if isinstance(expr, ast.Constant):
                return str(expr.value)
            if isinstance(expr, ast.Attribute):
                return expr.attr
            if isinstance(expr, ast.Tuple):
                parts = []
                for elt in expr.elts:
                    if isinstance(elt, ast.Attribute):
                        parts.append(elt.attr)
                return ",".join(parts)
            if isinstance(expr, ast.JoinedStr):
                return "".join(
                    p.value if isinstance(p, ast.Constant) else ""
                    for p in expr.values
                )
            return None

        slug_raw = fields.get("slug")
        slug: str | None = None
        if isinstance(slug_raw, ast.Constant):
            slug = str(slug_raw.value)
        elif isinstance(slug_raw, ast.Attribute):
            node: ast.expr = slug_raw
            while isinstance(node, ast.Attribute) and node.attr == "value":
                node = node.value
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "IntegrationSlug":
                    slug = node.attr.lower()
            elif isinstance(node, ast.Name):
                slug = node.id.lower()

        cats_expr = fields.get("categories")
        categories: list[str] = []
        if isinstance(cats_expr, ast.Tuple):
            for elt in cats_expr.elts:
                if isinstance(elt, ast.Attribute):
                    categories.append(elt.attr)

        if not slug or not categories:
            return None

        status = "STABLE"
        st = fields.get("status")
        if isinstance(st, ast.Attribute):
            status = st.attr

        env_prefix = '"INTERGRAX_"'
        ep = fields.get("env_prefix")
        if isinstance(ep, ast.Constant):
            env_prefix = repr(str(ep.value))

        description = '""'
        desc = fields.get("description")
        if isinstance(desc, ast.Constant):
            description = repr(str(desc.value))
        elif isinstance(desc, ast.JoinedStr):
            description = repr(_lit("description") or "")
        elif isinstance(desc, ast.Tuple):
            parts = [elt.value for elt in desc.elts if isinstance(elt, ast.Constant)]
            description = repr(" ".join(str(p) for p in parts))

        return {
            "slug": slug,
            "categories": ",".join(categories),
            "status": status,
            "env_prefix": env_prefix,
            "description": description,
        }
    return None


def _factory_import(text: str) -> str | None:
    for line in text.splitlines():
        if "from " in line and ".bundle import " in line:
            return line.strip()
        if "from " in line and " import create_" in line:
            return line.strip()
    return None


def _register_func_name(text: str) -> str | None:
    m = re.search(r"def\s+(register_\w+)\s*\(", text)
    return m.group(1) if m else None


def _relative_provider(register_path: Path) -> str:
    rel = register_path.parent.relative_to(PROVIDERS)
    return str(rel).replace("\\", ".")


def _write_manifest(path: Path, meta: dict[str, str]) -> None:
    cats = ", ".join(f"IntegrationCategory.{c}" for c in meta["categories"].split(","))
    content = f'''# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``{meta["slug"]}`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="{meta["slug"]}",
    categories=({cats},),
    status=IntegrationStatus.{meta["status"]},
    env_prefix={meta["env_prefix"]},
    description={meta["description"]},
)
'''
    path.write_text(content, encoding="utf-8")


def migrate_one(path: Path, *, dry_run: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    if "register_from_manifest" in text and "IntegrationEntry" not in text:
        return False
    meta = _extract_via_ast(text)
    if not meta:
        print(f"SKIP (parse): {path}", file=sys.stderr)
        return False
    factory_line = _factory_import(text)
    if not factory_line:
        print(f"SKIP (no factory): {path}", file=sys.stderr)
        return False
    factory_name = factory_line.split()[-1].rstrip(",)")
    func_name = _register_func_name(text) or f"register_{meta['slug']}_integration"
    rel = _relative_provider(path)

    if dry_run:
        print(f"OK {meta['slug']}: {rel}")
        return True

    _write_manifest(path.parent / "manifest.py", meta)
    content = f'''# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register {meta["slug"]} in the integration catalog."""

from __future__ import annotations

{factory_line}
from intergrax.integrations.providers.{rel}.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def {func_name}(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, {factory_name}, override=override)
'''
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    updated = 0
    for path in _find_register_files():
        if migrate_one(path, dry_run=dry_run):
            updated += 1
    print(f"{'would update' if dry_run else 'updated'} {updated} providers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
