"""RW-BND-05 (NT-BND-05-1): Tier-2 agents must not construct RuntimeContext; posture from host (FC-1 Part C1).

Posture: a Tier-2 agent must not hard-code ``production_mode`` as a literal or read the lab constant
``harness_production_mode()``; the host injects posture (EQOA-A4-015b).

Construction: Tier-2 modules under ``agents/`` and ``intergrax/agents/`` must not call
``RuntimeContext(...)`` or ``RuntimeContext.build(...)``; the sanctioned host composes runtime context.
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
_RUNTIME_CONTEXT = "RuntimeContext"


@dataclass(frozen=True, slots=True)
class Violation:
    relative_path: str
    line: int
    reason: str


@dataclass(frozen=True, slots=True)
class ConstructionViolation:
    relative_path: str
    line: int
    kind: str


def _called_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def scan_posture_violations(relative_path: str, source: str) -> list[Violation]:
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


def _runtime_context_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names to ``RuntimeContext`` when imported or assigned from it."""
    aliases: dict[str, str] = {_RUNTIME_CONTEXT: _RUNTIME_CONTEXT}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == _RUNTIME_CONTEXT:
                    local = alias.asname or alias.name
                    aliases[local] = _RUNTIME_CONTEXT
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if not isinstance(node.value, ast.Name):
                continue
            rhs = node.value.id
            if aliases.get(rhs) == _RUNTIME_CONTEXT:
                aliases[node.targets[0].id] = _RUNTIME_CONTEXT
    return aliases


def _resolves_runtime_context_name(name: str, aliases: dict[str, str]) -> bool:
    return aliases.get(name) == _RUNTIME_CONTEXT or name == _RUNTIME_CONTEXT


def scan_runtime_context_construction(relative_path: str, source: str) -> list[ConstructionViolation]:
    """Report direct or static ``build`` construction of ``RuntimeContext`` in a module."""
    tree = ast.parse(source)
    aliases = _runtime_context_aliases(tree)
    violations: list[ConstructionViolation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if _resolves_runtime_context_name(func.id, aliases):
                violations.append(
                    ConstructionViolation(relative_path, node.lineno, "unauthorized RuntimeContext construction")
                )
            continue
        if not isinstance(func, ast.Attribute) or func.attr != "build":
            continue
        value = func.value
        if isinstance(value, ast.Name) and _resolves_runtime_context_name(value.id, aliases):
            violations.append(
                ConstructionViolation(relative_path, node.lineno, "unauthorized RuntimeContext.build call")
            )
        elif isinstance(value, ast.Attribute) and value.attr == _RUNTIME_CONTEXT:
            violations.append(
                ConstructionViolation(relative_path, node.lineno, "unauthorized RuntimeContext.build call")
            )

    return violations


def test_tier2_build_context_takes_posture_from_host_not_a_literal() -> None:
    violations: list[Violation] = []
    for root in _TIER2_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            relative = path.relative_to(_REPO)
            if _EXCLUDED_PARTS.intersection(relative.parts):
                continue
            violations.extend(scan_posture_violations(relative.as_posix(), path.read_text(encoding="utf-8")))
    assert violations == [], "\n".join(f"{v.relative_path}:{v.line}: {v.reason}" for v in violations)


def test_scan_flags_a_rogue_agent() -> None:
    """Self-test: construction and posture scans report seeded rogue patterns; controls stay clean."""
    rogue_construction = (
        "from intergrax.runtime.execution.agent_runtime_context import RuntimeContext\n"
        "class Rogue:\n"
        "    def build_context(self, request):\n"
        "        a = RuntimeContext(config=x, session_manager=y)\n"
        "        b = RuntimeContext.build(config=x, session_manager=y)\n"
        "        return a, b\n"
    )
    kinds = sorted(v.kind for v in scan_runtime_context_construction("agents/rogue/rogue_agent.py", rogue_construction))
    assert kinds == [
        "unauthorized RuntimeContext construction",
        "unauthorized RuntimeContext.build call",
    ]

    injected = "class Ok:\n    def run(self, runtime_ctx):\n        return runtime_ctx\n"
    assert scan_runtime_context_construction("agents/ok/ok_agent.py", injected) == []

    annotation_only = (
        "from intergrax.runtime.execution.agent_runtime_context import RuntimeContext\n"
        "def annotate(ctx: RuntimeContext) -> None:\n"
        "    pass\n"
    )
    assert scan_runtime_context_construction("agents/ok/ok_agent.py", annotation_only) == []

    unrelated_build = "class Demo:\n    def f(self):\n        return self.registry.build(id='x')\n"
    assert scan_runtime_context_construction("agents/ok/ok_agent.py", unrelated_build) == []

    import_alias = (
        "from intergrax.runtime.execution.agent_runtime_context import RuntimeContext as RC\n"
        "class Rogue:\n"
        "    def build_context(self, request):\n"
        "        return RC.build(config=c, session_manager=s)\n"
    )
    assert [v.kind for v in scan_runtime_context_construction("agents/rogue/alias.py", import_alias)] == [
        "unauthorized RuntimeContext.build call"
    ]

    rogue_posture = (
        "class Rogue:\n"
        "    def build_context(self, request):\n"
        "        a = RuntimeConfig(llm_adapter=x, production_mode=False)\n"
        "        b = RuntimeConfig(llm_adapter=x, production_mode=harness_production_mode())\n"
        "        return a, b\n"
    )
    reasons = sorted(v.reason for v in scan_posture_violations("agents/rogue/rogue_agent.py", rogue_posture))
    assert reasons == ["production_mode is a literal", "production_mode reads the lab constant"]
    injected_posture = "config = RuntimeConfig(llm_adapter=x, production_mode=self._production_mode)\n"
    assert scan_posture_violations("agents/ok/ok_agent.py", injected_posture) == []


_ARCHITECT_CENSUS_SITES: tuple[tuple[str, str], ...] = (
    ("agents/boundary_demo/boundary_demo_agent.py", "unauthorized RuntimeContext.build call"),
    (
        "agents/external_contractor_adapter/external_contractor_adapter_agent.py",
        "unauthorized RuntimeContext.build call",
    ),
    ("agents/organization_worker/organization_worker_agent.py", "unauthorized RuntimeContext.build call"),
    ("intergrax/agents/authoring/acp_stub_reflex.py", "unauthorized RuntimeContext.build call"),
)


@pytest.mark.parametrize(("relative_path", "expected_kind"), _ARCHITECT_CENSUS_SITES)
def test_architect_census_sites_show_unauthorized_construction(relative_path: str, expected_kind: str) -> None:
    """Documented product residuals: detector sees real construction sites (not masked in the test)."""
    path = _REPO / relative_path
    text = path.read_text(encoding="utf-8")
    kinds = {v.kind for v in scan_runtime_context_construction(relative_path, text)}
    assert expected_kind in kinds
