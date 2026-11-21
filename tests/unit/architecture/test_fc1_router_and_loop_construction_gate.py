"""FC-1 construction gate: every router and loop construction states its posture.

NT-FC1-12 of T-IGX-043-A (EQOA-A4-044, A4-015a): product code constructs ``NexusLoop`` and
``AgentRouter`` only with an explicit ``production_mode=``, ``GraphExecutor`` only with ``router=``,
and ``AgentRouter`` only at the two sanctioned sites. The gate has a self-test on a rogue source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PRODUCT_ROOTS = ("intergrax", "agents", "applications")
_EXCLUDED_PARTS = frozenset({"tests", "notebooks", "__pycache__", ".venv", "node_modules"})
_ALLOWED_ROUTER_SITES = frozenset(
    {
        "intergrax/runtime/nexus/nexus_loop.py",
        "intergrax/runtime/nexus/execution/graph_executor.py",
    },
)
_ALLOWED_GRAPH_EXECUTOR_SITES = frozenset({"intergrax/runtime/nexus/nexus_loop.py"})


@dataclass(frozen=True, slots=True)
class Violation:
    relative_path: str
    line: int
    rule: str


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def scan_source(relative_path: str, source: str) -> list[Violation]:
    """Return the construction-posture violations of one module source."""
    violations: list[Violation] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        keywords = {kw.arg for kw in node.keywords}
        if name in {"NexusLoop", "AgentRouter"} and "production_mode" not in keywords:
            violations.append(Violation(relative_path, node.lineno, f"{name} without production_mode"))
        if name == "AgentRouter" and relative_path not in _ALLOWED_ROUTER_SITES:
            violations.append(Violation(relative_path, node.lineno, "AgentRouter outside the sanctioned sites"))
        if name == "GraphExecutor":
            if "router" not in keywords:
                violations.append(Violation(relative_path, node.lineno, "GraphExecutor without router"))
            if relative_path not in _ALLOWED_GRAPH_EXECUTOR_SITES:
                violations.append(Violation(relative_path, node.lineno, "GraphExecutor outside nexus_loop"))
    return violations


def _product_sources() -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for root in _PRODUCT_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            relative = path.relative_to(_REPO)
            if _EXCLUDED_PARTS.intersection(relative.parts):
                continue
            sources.append((relative.as_posix(), path.read_text(encoding="utf-8")))
    return sources


def test_no_product_construction_without_explicit_posture() -> None:
    """NT-FC1-12."""
    violations = [v for rel, src in _product_sources() for v in scan_source(rel, src)]
    assert violations == [], "\n".join(f"{v.relative_path}:{v.line}: {v.rule}" for v in violations)


def test_gate_flags_a_rogue_construction() -> None:
    """Self-test: the scan reports the constructions it is meant to forbid."""
    rogue = (
        "from intergrax.runtime.nexus.agent_router import AgentRouter\n"
        "from intergrax.runtime.nexus.nexus_loop import NexusLoop\n"
        "router = AgentRouter(registry)\n"
        "loop = NexusLoop(registry)\n"
        "executor = GraphExecutor(registry)\n"
    )
    rules = {v.rule for v in scan_source("intergrax/rogue_module.py", rogue)}
    assert rules == {
        "AgentRouter without production_mode",
        "AgentRouter outside the sanctioned sites",
        "NexusLoop without production_mode",
        "GraphExecutor without router",
        "GraphExecutor outside nexus_loop",
    }
    sanctioned = "router = AgentRouter(registry, production_mode=False)\n"
    assert scan_source("intergrax/runtime/nexus/execution/graph_executor.py", sanctioned) == []
