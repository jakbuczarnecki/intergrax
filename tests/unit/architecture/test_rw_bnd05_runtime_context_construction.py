"""RW-BND-05 (NT-BND-05-1): Tier-2 agents take the runtime posture from the host, not from a literal.

A Tier-2 agent that builds a ``RuntimeConfig`` must not hard-code ``production_mode`` (a literal) or
read the lab constant ``harness_production_mode()``; the posture is injected by the composing host
(FC-1 Part C1, EQOA-A4-015b). The scan has no allowlist and a self-test on a rogue source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_TIER2_ROOTS = ("agents", "intergrax/agents")
_EXCLUDED_PARTS = frozenset({"tests", "notebooks", "__pycache__"})


@dataclass(frozen=True, slots=True)
class Violation:
    relative_path: str
    line: int
    reason: str


def _called_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def scan_source(relative_path: str, source: str) -> list[Violation]:
    """Return every ``RuntimeConfig(...)`` whose ``production_mode`` is a literal or the lab constant."""
    violations: list[Violation] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or _called_name(node) != "RuntimeConfig":
            continue
        for keyword in node.keywords:
            if keyword.arg != "production_mode":
                continue
            if isinstance(keyword.value, ast.Constant):
                violations.append(Violation(relative_path, node.lineno, "production_mode is a literal"))
            elif _called_name(keyword.value) == "harness_production_mode":
                violations.append(Violation(relative_path, node.lineno, "production_mode reads the lab constant"))
    return violations


def test_tier2_build_context_takes_posture_from_host_not_a_literal() -> None:
    violations: list[Violation] = []
    for root in _TIER2_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            relative = path.relative_to(_REPO)
            if _EXCLUDED_PARTS.intersection(relative.parts):
                continue
            violations.extend(scan_source(relative.as_posix(), path.read_text(encoding="utf-8")))
    assert violations == [], "\n".join(f"{v.relative_path}:{v.line}: {v.reason}" for v in violations)


def test_scan_flags_a_rogue_agent() -> None:
    """Self-test: literal and lab-constant postures are reported; an injected posture is not."""
    rogue = (
        "class Rogue:\n"
        "    def build_context(self, request):\n"
        "        a = RuntimeConfig(llm_adapter=x, production_mode=False)\n"
        "        b = RuntimeConfig(llm_adapter=x, production_mode=harness_production_mode())\n"
        "        return a, b\n"
    )
    reasons = sorted(v.reason for v in scan_source("agents/rogue/rogue_agent.py", rogue))
    assert reasons == ["production_mode is a literal", "production_mode reads the lab constant"]
    injected = "config = RuntimeConfig(llm_adapter=x, production_mode=self._production_mode)\n"
    assert scan_source("agents/ok/ok_agent.py", injected) == []
