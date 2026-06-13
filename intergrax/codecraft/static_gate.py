# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""StaticCodeGate — L0 AST/import/size scan before sandbox exec (ECC-1)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from intergrax.codecraft.contracts import StaticGateResult
from intergrax.codecraft.profile import CodeCraftProfile, DEFAULT_FORBIDDEN_IMPORTS

_FORBIDDEN_CALL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("forbidden_call_eval", re.compile(r"\beval\s*\(")),
    ("forbidden_call_exec", re.compile(r"\bexec\s*\(")),
    ("forbidden_call_compile", re.compile(r"\bcompile\s*\(")),
    ("forbidden_dunder_import", re.compile(r"__import__\s*\(")),
)


@dataclass(frozen=True)
class StaticCodeGate:
    """Scan candidate code before any sandbox execution."""

    profile: CodeCraftProfile

    def scan(self, code: str, *, language: str = "python") -> StaticGateResult:
        if language.strip().lower() not in {lang.lower() for lang in self.profile.allowed_languages}:
            return StaticGateResult(
                passed=False,
                rule_ids=["language_not_allowed"],
                message=f"language {language!r} not in allowed_languages",
            )

        encoded = code.encode("utf-8")
        if len(encoded) > self.profile.max_code_bytes:
            return StaticGateResult(
                passed=False,
                rule_ids=["code_size_exceeded"],
                message=f"code exceeds max_code_bytes ({len(encoded)} > {self.profile.max_code_bytes})",
            )

        if language.strip().lower() != "python":
            return StaticGateResult(passed=True, rule_ids=[])

        for rule_id, pattern in _FORBIDDEN_CALL_PATTERNS:
            if pattern.search(code):
                return StaticGateResult(
                    passed=False,
                    rule_ids=[rule_id],
                    message=f"forbidden pattern: {rule_id}",
                )

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return StaticGateResult(
                passed=False,
                rule_ids=["syntax_error"],
                message=str(exc),
            )

        forbidden = {item.lower() for item in self.profile.forbidden_imports or DEFAULT_FORBIDDEN_IMPORTS}
        violations = _collect_import_violations(tree, forbidden)
        if violations:
            return StaticGateResult(
                passed=False,
                rule_ids=["forbidden_import"],
                message=f"forbidden imports: {', '.join(sorted(violations))}",
            )

        return StaticGateResult(passed=True, rule_ids=[])


def _collect_import_violations(tree: ast.AST, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".")[0].lower()
                if root in forbidden:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0].lower()
                if root in forbidden:
                    found.add(root)
    return found
