#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail CI when Tier-2 agents import integrations/tools directly (Phase H-APP.0.4)."""

from __future__ import annotations

import ast
from pathlib import Path

SCAN_ROOT = "agents"
ALLOWLIST = frozenset({"agents/__init__.py"})
_CONTRACT_PREFIX = "intergrax.integrations.contracts"
_REASON = "agents may import only intergrax.integrations.contracts leaves"


def _allowed_contract_module(module: str) -> bool:
    return module == _CONTRACT_PREFIX or module.startswith(f"{_CONTRACT_PREFIX}.")


def _import_name_violation(module: str) -> str | None:
    if module.startswith("intergrax.tools"):
        return _REASON
    if module.startswith("intergrax.integrations") and not _allowed_contract_module(module):
        return _REASON
    return None


def _import_from_violations(node: ast.ImportFrom) -> list[tuple[int, str]]:
    if node.level != 0:
        return []
    lineno = node.lineno
    findings: list[tuple[int, str]] = []
    if node.module is None:
        return []
    module = node.module
    if module == "intergrax":
        for alias in node.names:
            if alias.name in {"integrations", "tools"}:
                findings.append((lineno, _REASON))
        return findings
    violation = _import_name_violation(module)
    if violation is not None:
        findings.append((lineno, violation))
    return findings


def _collect_violations(tree: ast.AST) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _import_name_violation(alias.name)
                if violation is not None:
                    findings.append((node.lineno, violation))
        elif isinstance(node, ast.ImportFrom):
            findings.extend(_import_from_violations(node))
    return findings


def check_imports_in_tree(root: Path) -> list[str]:
    """Scan ``root/agents/**/*.py`` for DP-14 import violations and parse failures."""
    agents_root = root / SCAN_ROOT
    if not agents_root.is_dir():
        return []

    findings: list[str] = []
    for path in sorted(agents_root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            findings.append(f"{rel}:0: read error: {exc}")
            continue
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            line = exc.lineno or 0
            findings.append(f"{rel}:{line}: syntax error: {exc.msg}")
            continue
        for lineno, reason in _collect_violations(tree):
            findings.append(f"{rel}:{lineno}: {reason}")
    return sorted(findings)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    findings = check_imports_in_tree(repo_root)
    if findings:
        print("Tier-2 agents must not import intergrax.integrations/tools directly:")
        for item in findings:
            print(f"  - {item}")
        return 1
    print("check_agent_registry_bypass: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
