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
_CANONICAL_MODULE = "intergrax.runtime.execution.agent_runtime_context"

# Diagnostic record (historical census); not an acceptance assertion on product code.
_DOCUMENTED_UNAUTHORIZED_CENSUS_SITES: tuple[tuple[str, int, str], ...] = (
    ("agents/boundary_demo/boundary_demo_agent.py", 96, "unauthorized RuntimeContext.build call"),
    (
        "agents/external_contractor_adapter/external_contractor_adapter_agent.py",
        112,
        "unauthorized RuntimeContext.build call",
    ),
    ("agents/organization_worker/organization_worker_agent.py", 118, "unauthorized RuntimeContext.build call"),
    ("intergrax/agents/authoring/acp_stub_reflex.py", 69, "unauthorized RuntimeContext.build call"),
)


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


def _dotted_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted_name(expr.value)
        if base is None:
            return None
        return f"{base}.{expr.attr}"
    return None


def _is_canonical_runtime_context_module(dotted: str) -> bool:
    return dotted == _CANONICAL_MODULE or dotted.endswith(".agent_runtime_context")


def _register_module_alias(aliases: dict[str, str], local: str, imported: str) -> None:
    if _is_canonical_runtime_context_module(imported) or imported == _CANONICAL_MODULE:
        aliases[local] = _CANONICAL_MODULE


def _runtime_context_bindings(tree: ast.AST) -> tuple[frozenset[str], dict[str, str]]:
    """Map local names to the canonical ``RuntimeContext`` class and its module."""
    class_aliases: set[str] = set()
    module_aliases: dict[str, str] = {}
    module_level_shadowed = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[-1]
                if alias.name == _CANONICAL_MODULE or alias.name.endswith(".agent_runtime_context"):
                    _register_module_alias(module_aliases, local, alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                continue
            module_path = node.module
            for alias in node.names:
                if alias.name == "agent_runtime_context":
                    local = alias.asname or alias.name
                    if _is_canonical_runtime_context_module(module_path):
                        _register_module_alias(module_aliases, local, _CANONICAL_MODULE)
                if module_path == _CANONICAL_MODULE and alias.name == _RUNTIME_CONTEXT:
                    local = alias.asname or alias.name
                    class_aliases.add(local)
        elif isinstance(node, ast.ClassDef) and node.name == _RUNTIME_CONTEXT and node.col_offset == 0:
            module_level_shadowed = True

    if module_level_shadowed:
        class_aliases.discard(_RUNTIME_CONTEXT)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if isinstance(node.value, ast.Name) and node.value.id in class_aliases:
            class_aliases.add(target)

    return frozenset(class_aliases), module_aliases


def _attribute_is_canonical_runtime_class(expr: ast.expr, module_aliases: dict[str, str]) -> bool:
    if not isinstance(expr, ast.Attribute) or expr.attr != _RUNTIME_CONTEXT:
        return False
    if isinstance(expr.value, ast.Name) and expr.value.id in module_aliases:
        return True
    dotted = _dotted_name(expr.value)
    return dotted is not None and _is_canonical_runtime_context_module(dotted)


def _refers_to_canonical_runtime_context(
    expr: ast.expr,
    class_aliases: frozenset[str],
    module_aliases: dict[str, str],
) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in class_aliases
    if _attribute_is_canonical_runtime_class(expr, module_aliases):
        return True
    if isinstance(expr, ast.Attribute) and expr.attr == "build":
        return _refers_to_canonical_runtime_context(expr.value, class_aliases, module_aliases)
    dotted = _dotted_name(expr)
    return dotted == f"{_CANONICAL_MODULE}.{_RUNTIME_CONTEXT}"


def _classify_runtime_context_call(
    func: ast.expr,
    class_aliases: frozenset[str],
    module_aliases: dict[str, str],
) -> str | None:
    if isinstance(func, ast.Name) and func.id in class_aliases:
        return "unauthorized RuntimeContext construction"
    if isinstance(func, ast.Attribute):
        if func.attr == _RUNTIME_CONTEXT and _attribute_is_canonical_runtime_class(func, module_aliases):
            return "unauthorized RuntimeContext construction"
        if func.attr == "build" and _refers_to_canonical_runtime_context(func.value, class_aliases, module_aliases):
            return "unauthorized RuntimeContext.build call"
    return None


def scan_runtime_context_construction(relative_path: str, source: str) -> list[ConstructionViolation]:
    """Report direct or static ``build`` construction of canonical ``RuntimeContext`` in a module."""
    tree = ast.parse(source)
    class_aliases, module_aliases = _runtime_context_bindings(tree)
    violations: list[ConstructionViolation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kind = _classify_runtime_context_call(node.func, class_aliases, module_aliases)
        if kind is not None:
            violations.append(ConstructionViolation(relative_path, node.lineno, kind))

    return violations


def iter_tier2_python_sources() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for root in _TIER2_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            relative = path.relative_to(_REPO)
            if _EXCLUDED_PARTS.intersection(relative.parts):
                continue
            paths.append((relative.as_posix(), path))
    return paths


def collect_tier2_runtime_context_construction_violations() -> list[ConstructionViolation]:
    violations: list[ConstructionViolation] = []
    for relative, path in iter_tier2_python_sources():
        violations.extend(scan_runtime_context_construction(relative, path.read_text(encoding="utf-8")))
    return sorted(violations, key=lambda v: (v.relative_path, v.line, v.kind))


def test_tier2_build_context_takes_posture_from_host_not_a_literal() -> None:
    violations: list[Violation] = []
    for relative, path in iter_tier2_python_sources():
        violations.extend(scan_posture_violations(relative, path.read_text(encoding="utf-8")))
    assert violations == [], "\n".join(f"{v.relative_path}:{v.line}: {v.reason}" for v in violations)


def _assert_detector_synthetic_controls() -> None:
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
        "        direct = RC(config=c, session_manager=s)\n"
        "        return RC.build(config=c, session_manager=s), direct\n"
    )
    alias_kinds = sorted(v.kind for v in scan_runtime_context_construction("agents/rogue/alias.py", import_alias))
    assert alias_kinds == [
        "unauthorized RuntimeContext construction",
        "unauthorized RuntimeContext.build call",
    ]

    module_alias = (
        "import intergrax.runtime.execution.agent_runtime_context as arc\n"
        "def rogue():\n"
        "    return arc.RuntimeContext.build(config=c, session_manager=s)\n"
    )
    assert [v.kind for v in scan_runtime_context_construction("agents/rogue/module_alias.py", module_alias)] == [
        "unauthorized RuntimeContext.build call"
    ]

    qualified = (
        "def rogue():\n"
        "    return intergrax.runtime.execution.agent_runtime_context.RuntimeContext("
        "config=a, session_manager=b)\n"
    )
    assert scan_runtime_context_construction("agents/rogue/qualified.py", qualified) == [
        ConstructionViolation("agents/rogue/qualified.py", 2, "unauthorized RuntimeContext construction")
    ]

    unrelated_class = (
        "class RuntimeContext:\n"
        "    @classmethod\n"
        "    def build(cls, **kwargs):\n"
        "        return cls()\n"
        "def rogue():\n"
        "    return RuntimeContext.build(config=1, session_manager=2)\n"
    )
    assert scan_runtime_context_construction("agents/rogue/local_class.py", unrelated_class) == []

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


def test_detector_synthetic_runtime_context_construction_controls() -> None:
    """Synthetic controls: detector behaviour must stay green after product fixes."""
    _assert_detector_synthetic_controls()


def test_scan_flags_a_rogue_agent() -> None:
    """FRZ-BND-05 canonical node: synthetic detector controls and Tier-2 product gate."""
    _assert_detector_synthetic_controls()
    violations = collect_tier2_runtime_context_construction_violations()
    assert violations == [], "\n".join(f"{v.relative_path}:{v.line}: {v.kind}" for v in violations)
