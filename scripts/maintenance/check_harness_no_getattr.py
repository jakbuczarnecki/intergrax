#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when getattr/setattr appear outside approved bridge modules (AST call sites only)."""

from __future__ import annotations

import ast
from pathlib import Path

SCAN_ROOTS = (
    "intergrax",
    "agents",
    "tests",
    "scripts",
    "testing_support",
)

# PR / pre-push smoke: Tier-2 agents only (ci_smoke pytest covers wider architecture gates).
SMOKE_SCAN_ROOTS = ("agents",)

GRANDFATHER: frozenset[str] = frozenset(
    {
        "intergrax/utils/attribute_access.py",
        "intergrax/utils/lazy_export.py",
        "scripts/codemods/codemod_remove_getattr.py",
        "scripts/codemods/codemod_remove_getattr_text.py",
        "scripts/codemods/patch_lazy_bundle_exports.py",
        "scripts/maintenance/add_provider_protocol_delegates.py",
        "scripts/maintenance/cutover_provider_runtime_integrations.py",
        "scripts/maintenance/fix_runtime_delegation.py",
        "tests/unit/integrations/providers/test_provider_category_contract_migration.py",
        "tests/unit/integrations/providers/test_provider_legacy_delegation_removed.py",
        "tests/unit/integrations/providers/test_provider_runtime_cutover.py",
        "tests/unit/integrations/providers/observability_backend/test_observability_legacy_delegation_removed.py",
        "tests/unit/integrations/providers/observability_backend/test_observability_provider_contract_migration.py",
        "tests/unit/runtime/integrations/test_provider_category_contracts.py",
        "tests/unit/runtime/token_optimization/test_signals.py",
        "tests/unit/hosting/signals/test_signal_adapter.py",
        "tests/unit/applications/local_workspace_application/test_lkw_core_platform_proof.py",
        "tools/ast_audit.py",
    }
)

_REFLECTION_NAMES = frozenset({"getattr", "setattr"})


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan_source(rel: str, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return [f"{rel}:{exc.lineno or 0}: syntax error ({exc.msg})"]
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name not in _REFLECTION_NAMES:
            continue
        if name == "setattr" and _is_monkeypatch_setattr(node, source):
            continue
        line = source.splitlines()[node.lineno - 1].strip() if node.lineno else name
        violations.append(f"{rel}:{node.lineno}: {line}")
    return violations


def _is_monkeypatch_setattr(node: ast.Call, source: str) -> bool:
    if node.lineno is None:
        return False
    line = source.splitlines()[node.lineno - 1]
    return "monkeypatch.setattr" in line or "mp.setattr" in line


def _scan_roots(roots: tuple[str, ...]) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    violations: list[str] = []
    for root_name in roots:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root).as_posix()
            if rel in GRANDFATHER:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            violations.extend(_scan_source(rel, source))
    return violations


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AST getattr/setattr gate.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Scan intergrax/, agents/, scripts/ only (PR/pre-push default).",
    )
    args = parser.parse_args()
    roots = SMOKE_SCAN_ROOTS if args.smoke else SCAN_ROOTS
    violations = _scan_roots(roots)
    if violations:
        label = "smoke" if args.smoke else "full"
        print(f"getattr/setattr call violations ({label} scan, outside bridge modules):")
        print("\n".join(sorted(violations)))
        return 1
    label = "smoke" if args.smoke else "full"
    print(f"getattr/setattr audit: OK ({label} scan, bridge modules only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
