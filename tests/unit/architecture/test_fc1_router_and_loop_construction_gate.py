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
_PROTECTED_CONSTRUCTIONS = frozenset({"NexusLoop", "AgentRouter", "GraphExecutor"})
_ALLOWED_ROUTER_SITES = frozenset(
    {
        "intergrax/runtime/nexus/nexus_loop.py",
        "intergrax/runtime/nexus/execution/graph_executor.py",
    },
)
_ALLOWED_GRAPH_EXECUTOR_SITES = frozenset({"intergrax/runtime/nexus/nexus_loop.py"})
_SANCTIONED_PRODUCTION_CALLERS = (
    "intergrax/runtime/nexus/nexus_loop.py",
    "intergrax/runtime/nexus/execution/graph_executor.py",
    "intergrax/runtime/execution/environment_orchestration_materialization.py",
    "intergrax/runtime/execution/debug_lab_nexus_loop.py",
)


def _is_application_docker_runtime_context_mirror(parts: tuple[str, ...]) -> bool:
    """Distribution mirrors are vendored copies, not composition roots (W3 C1-R)."""
    return len(parts) >= 4 and parts[0] == "applications" and parts[2] == "docker" and parts[3] == "runtime-context"


@dataclass(frozen=True, slots=True)
class Violation:
    relative_path: str
    line: int
    rule: str


def _build_constructor_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names to canonical protected construction symbols."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _PROTECTED_CONSTRUCTIONS:
                    local = alias.asname or alias.name
                    aliases[local] = alias.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if not isinstance(node.value, ast.Name):
                continue
            rhs = node.value.id
            canonical = aliases.get(rhs, rhs)
            if canonical in _PROTECTED_CONSTRUCTIONS:
                aliases[node.targets[0].id] = canonical
    return aliases


def _resolved_construction_name(node: ast.Call, aliases: dict[str, str]) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        local = func.id
        if local in aliases:
            return aliases[local]
        if local in _PROTECTED_CONSTRUCTIONS:
            return local
        return None
    if isinstance(func, ast.Attribute) and func.attr in _PROTECTED_CONSTRUCTIONS:
        return func.attr
    return None


def scan_source(relative_path: str, source: str) -> list[Violation]:
    """Return the construction-posture violations of one module source."""
    tree = ast.parse(source)
    aliases = _build_constructor_aliases(tree)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _resolved_construction_name(node, aliases)
        if name is None:
            continue
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
            if _is_application_docker_runtime_context_mirror(relative.parts):
                continue
            sources.append((relative.as_posix(), path.read_text(encoding="utf-8")))
    return sources


def test_no_product_construction_without_explicit_posture() -> None:
    """NT-FC1-12."""
    violations = [v for rel, src in _product_sources() for v in scan_source(rel, src)]
    assert violations == [], "\n".join(f"{v.relative_path}:{v.line}: {v.rule}" for v in violations)


def test_unrelated_class_constructors_are_not_banned() -> None:
    """The gate targets only the FC-1 protected trio, not every ``__init__``/call."""
    source = "class DemoRegistry:\n    pass\nreg = DemoRegistry()\n"
    assert scan_source("intergrax/example/demo_registry.py", source) == []


def test_direct_constructor_without_posture_is_forbidden() -> None:
    source = "from intergrax.runtime.nexus.nexus_loop import NexusLoop\nloop = NexusLoop(registry)\n"
    rules = {v.rule for v in scan_source("intergrax/rogue/direct_nexus.py", source)}
    assert rules == {"NexusLoop without production_mode"}


def test_approved_factory_construction_passes_scan() -> None:
    """Approved EE materialization pattern: explicit posture from the orchestration spec."""
    source = (
        "from intergrax.runtime.nexus.nexus_loop import NexusLoop\n"
        "def materialize_host_orchestration_backend(registry, spec):\n"
        "    return NexusLoop(registry, production_mode=spec.production_mode)\n"
    )
    rel = "intergrax/runtime/execution/environment_orchestration_materialization.py"
    assert scan_source(rel, source) == []


@pytest.mark.parametrize("relative_path", _SANCTIONED_PRODUCTION_CALLERS)
def test_sanctioned_production_callers_pass(relative_path: str) -> None:
    path = _REPO / relative_path
    violations = scan_source(relative_path, path.read_text(encoding="utf-8"))
    assert violations == [], "\n".join(f"{v.line}: {v.rule}" for v in violations)


def test_constructor_import_alias_without_posture_is_forbidden() -> None:
    source = "from intergrax.runtime.nexus.nexus_loop import NexusLoop as LoopCtor\nloop = LoopCtor(registry)\n"
    rules = {v.rule for v in scan_source("intergrax/rogue/alias_import.py", source)}
    assert rules == {"NexusLoop without production_mode"}


def test_constructor_assign_alias_without_posture_is_forbidden() -> None:
    source = (
        "from intergrax.runtime.nexus.agent_router import AgentRouter\n"
        "Router = AgentRouter\n"
        "router = Router(registry)\n"
    )
    rules = {v.rule for v in scan_source("intergrax/rogue/alias_assign.py", source)}
    assert rules == {
        "AgentRouter without production_mode",
        "AgentRouter outside the sanctioned sites",
    }


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
