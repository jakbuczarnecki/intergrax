#!/usr/bin/env python3
"""Patch complex provider __init__.py with contract integration lazy exports."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys_path = ROOT
import sys

sys.path.insert(0, str(sys_path))

from scripts.maintenance.migrate_provider_contract_integrations import (  # noqa: E402
    class_prefix,
    contract_factory_name,
    provider_id_const,
)

PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"


def _import_base(path: Path) -> str:
    rel = path.parent.relative_to(ROOT / "intergrax")
    return "intergrax." + ".".join(rel.parts)


def _slug_category(path: Path) -> tuple[str, str]:
    slug = path.parent.name
    category = path.parent.parent.name
    return slug, category


def patch_init(path: Path) -> bool:
    slug, category = _slug_category(path)
    if not (path.parent / "integration.py").is_file():
        return False

    text = path.read_text(encoding="utf-8")
    prefix = class_prefix(slug, category)
    const = provider_id_const(slug, category)
    contract_factory = contract_factory_name(slug, category)
    import_base = _import_base(path)

    integration_exports = [
        const,
        f"{prefix}Integration",
        f"{prefix}IntegrationConfig",
        f"{prefix}Client",
    ]
    contract_block = (
        f"\n_CONTRACT_INTEGRATION_EXPORTS = frozenset(\n    {{\n"
        + "".join(f'        "{symbol}",\n' for symbol in integration_exports)
        + "    }\n)\n"
    )

    if "_CONTRACT_INTEGRATION_EXPORTS" not in text:
        text = re.sub(r"\ndef __getattr__", contract_block + "\ndef __getattr__", text, count=1)

    for symbol in integration_exports + [contract_factory]:
        if symbol not in text:
            text = re.sub(
                r"(__all__\s*=\s*\[)([^\]]*)",
                lambda m, s=symbol: m.group(1) + m.group(2).rstrip() + f'\n    "{s}",\n',
                text,
                count=1,
            )

    if contract_factory not in text and "_LAZY_EXPORTS" in text:
        text = re.sub(
            r"(_LAZY_EXPORTS = frozenset\(\s*\{)([^}]*)(\})",
            lambda m: m.group(1) + m.group(2).rstrip() + f'\n        "{contract_factory}",\n    ' + m.group(3),
            text,
            count=1,
        )

    if contract_factory not in text and "_BUNDLE_EXPORTS" in text:
        text = re.sub(
            r"(_BUNDLE_EXPORTS = frozenset\(\s*\{)([^}]*)(\})",
            lambda m: m.group(1) + m.group(2).rstrip() + f'\n        "{contract_factory}",\n    ' + m.group(3),
            text,
            count=1,
        )

    if contract_factory not in text and "_EXPORTS = {" in text:
        text = text.replace(
            "_EXPORTS = {",
            f'_EXPORTS = {{\n    "{contract_factory}": "{import_base}.bundle",',
            1,
        )

    handler = (
        "    if name in _CONTRACT_INTEGRATION_EXPORTS:\n"
        f"        from {import_base} import integration as _integration\n\n"
        "        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)\n"
    )
    if "if name in _CONTRACT_INTEGRATION_EXPORTS:" not in text:
        if "export_from_bundle" not in text:
            text = "from intergrax.utils.lazy_export import export_from_bundle\n" + text
        text = re.sub(r"(\n    raise AttributeError[^\n]*\n)", "\n" + handler + r"\1", text, count=1)

    # Contract integration class takes precedence over legacy adapter alias.
    for symbol in integration_exports:
        text = re.sub(
            rf'\n    if name == "{re.escape(symbol)}":\n'
            rf"        from .+ import {re.escape(symbol)}\n\n"
            rf"        return {re.escape(symbol)}\n",
            "",
            text,
        )

    path.write_text(text, encoding="utf-8")
    print(f"patched {path.relative_to(ROOT)}")
    return True


def main() -> None:
    count = 0
    for path in sorted(PROVIDERS.rglob("__init__.py")):
        if patch_init(path):
            count += 1
    print(f"patched {count} files")


if __name__ == "__main__":
    main()
