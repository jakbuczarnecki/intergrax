# © Artur Czarnecki. All rights reserved.

"""Inventory helpers for Tier-2 Agent.run contract drift in tests/support code."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from intergrax.agents.agent_contract import Agent

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SCAN_ROOTS = (
    _REPO_ROOT / "tests",
    _REPO_ROOT / "testing_support",
    _REPO_ROOT / "agents" / "lab",
)

# Intentional abstract-instantiation contract tests (not production doubles).
_ALLOWLIST: set[tuple[str, str]] = {
    (
        "tests/unit/architecture/test_ebh_2d_a_application_build_context_boundary.py",
        "test_incident_investigator_factory_requires_concrete_agent",
    ),
}


def _class_inherits_agent_directly(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Agent":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Agent":
            return True
    return False


def _class_defines_run(node: ast.ClassDef) -> bool:
    return any(
        isinstance(child, ast.AsyncFunctionDef) and child.name == "run"
        for child in node.body
    )


def find_direct_agent_subclasses_missing_run_ast() -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for root in _SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=rel)
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                if not _class_inherits_agent_directly(node):
                    continue
                if _class_defines_run(node):
                    continue
                violations.append((rel, node.name))
    return violations


def _class_inherits_name(node: ast.ClassDef, base_name: str) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == base_name:
            return True
        if isinstance(base, ast.Attribute) and base.attr == base_name:
            return True
    return False


def _class_defines_method(node: ast.ClassDef, method_name: str) -> bool:
    return any(
        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name
        for child in node.body
    )


def find_harness_reference_subclasses_missing_uaep_ast() -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for root in _SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=rel)
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                if not _class_inherits_name(node, "HarnessReferenceAgent"):
                    continue
                if _class_defines_method(node, "get_steps") and _class_defines_method(
                    node, "run_step"
                ):
                    continue
                violations.append((rel, node.name))
    return violations


def find_concrete_agent_subclasses_missing_run_runtime() -> list[str]:
    """Classes that are concrete but still expose abstract Agent.run."""
    violations: list[str] = []
    for rel, class_name in find_direct_agent_subclasses_missing_run_ast():
        key = (rel, class_name)
        if key in _ALLOWLIST:
            continue
        module_path = _REPO_ROOT / rel
        # Nested/local classes are skipped — gate focuses on module-level doubles.
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
        if not any(
            isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body
        ):
            continue
        spec_name = rel.replace("/", ".").removesuffix(".py")
        module = __import__(spec_name, fromlist=[class_name])
        cls = getattr(module, class_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        if not issubclass(cls, Agent) or cls is Agent:
            continue
        if inspect.isabstract(cls):
            continue
        if getattr(cls.run, "__isabstractmethod__", False):
            violations.append(f"{rel}:{class_name}")
    return violations
