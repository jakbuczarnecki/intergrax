#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when getattr/setattr appear outside approved bridge modules."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCAN_ROOTS = (
    "intergrax",
    "agents",
    "tests",
    "scripts",
    "testing_support",
)

GRANDFATHER: frozenset[str] = frozenset(
    {
        # --- utility bridge modules (safe getattr) ---
        "intergrax/utils/attribute_access.py",
        "intergrax/utils/lazy_export.py",
        # --- migration / codemod tools ---
        "scripts/codemods/codemod_remove_getattr.py",
        "scripts/codemods/codemod_remove_getattr_text.py",
        "scripts/codemods/patch_lazy_bundle_exports.py",
        "scripts/maintenance/add_provider_protocol_delegates.py",
        "scripts/maintenance/cutover_provider_runtime_integrations.py",
        "scripts/maintenance/fix_runtime_delegation.py",
        # --- test files (intentional module introspection) ---
        "tests/unit/integrations/providers/test_provider_category_contract_migration.py",
        "tests/unit/integrations/providers/test_provider_legacy_delegation_removed.py",
        "tests/unit/integrations/providers/test_provider_runtime_cutover.py",
        "tests/unit/integrations/providers/observability_backend/test_observability_legacy_delegation_removed.py",
        "tests/unit/integrations/providers/observability_backend/test_observability_provider_contract_migration.py",
        "tests/unit/runtime/integrations/test_provider_category_contracts.py",
        # --- AST audit tool ---
        "tools/ast_audit.py",
    }
)

PATTERN = re.compile(r"\b(getattr|setattr)\s*\(")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    violations: list[str] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root).as_posix()
            if rel in GRANDFATHER:
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if PATTERN.search(stripped):
                    if "monkeypatch.setattr" in stripped:
                        continue
                    violations.append(f"{rel}:{line_no}: {stripped}")
    if violations:
        print("getattr/setattr violations (outside bridge modules):")
        print("\n".join(violations))
        return 1
    print("getattr/setattr audit: OK (bridge modules only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
