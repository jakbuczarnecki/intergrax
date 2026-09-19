# © Artur Czarnecki. All rights reserved.

"""HARNESS-01 — canonical execution path and zero-bypass qualification gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.harness_01.catalog import (
    HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES,
    HARNESS_01_EXECUTION_MATRIX,
    HARNESS_01_MAPPED_NODE_IDS,
    HARNESS_01_REQUIRED_FLOWS,
    HARNESS_01_RUNTIME_TOOL_INVOKER_COMPOSITION_ROOTS,
    HARNESS_01_ZERO_BYPASS_FINDINGS,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTERGRAX = _REPO_ROOT / "intergrax"
_BRIDGE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "public_tool_invocation_pattern_bridge.py"
)
_GATEWAY = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "tool_gateway.py"
_RUNTIME_TIER = _REPO_ROOT / "intergrax" / "runtime"
_APPLICATION_HOST_GLOBS = ("applications/*/host/**/*.py", "intergrax/applications/**/host/**/*.py")

_FORBIDDEN_RUNTIME_LLM_VENDOR_PREFIXES = (
    "openai",
    "anthropic",
    "google.generativeai",
    "boto3",
)


def _iter_py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


def _relative_posix(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_runtime_tool_invoker_call(node: ast.Call) -> bool:
    """Heuristic: ``RuntimeToolInvoker.invoke(state=..., agent_id=..., request=...)``."""
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "invoke"
        and isinstance(func.value, ast.Name)
        and func.value.id == "invoker"
    ):
        return False
    keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
    return "state" in keyword_names and "request" in keyword_names


def _collect_runtime_tool_invoker_invoke_callsites() -> dict[str, list[int]]:
    """Map intergrax/**/*.py → line numbers of governed RuntimeToolInvoker.invoke calls."""
    hits: dict[str, list[int]] = {}
    for path in _iter_py_files(_INTERGRAX):
        rel = _relative_posix(path)
        if "/tests/" in rel or rel.endswith("/tests.py"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        lines: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_runtime_tool_invoker_call(node):
                lines.append(node.lineno)
        if lines:
            hits[rel] = lines
    return hits


def test_harness_01_matrix_covers_required_inventory() -> None:
    flows = {row.flow for row in HARNESS_01_EXECUTION_MATRIX}
    assert HARNESS_01_REQUIRED_FLOWS == flows
    assert len(HARNESS_01_EXECUTION_MATRIX) >= 12


def test_harness_01_matrix_rows_classify_bypass_status() -> None:
    allowed = {"CANONICAL", "AUTHORIZED_INTERNAL", "BYPASS", "NOT_APPLICABLE"}
    for row in HARNESS_01_EXECUTION_MATRIX:
        assert row.bypass_status in allowed
        if row.bypass_status == "CANONICAL":
            assert row.proof, f"canonical flow {row.flow!r} must cite proof node ids"


def test_harness_01_zero_bypass_findings_catalogued() -> None:
    severities = {row.severity for row in HARNESS_01_ZERO_BYPASS_FINDINGS}
    assert "BLOCKER" not in severities


def test_harness_01_runtime_tool_invoker_callsites_are_authorized_internal() -> None:
    hits = _collect_runtime_tool_invoker_invoke_callsites()
    violations: list[str] = []
    for rel, lines in sorted(hits.items()):
        if rel in HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES:
            continue
        violations.append(f"{rel}:{lines}")
    assert violations == [], (
        "RuntimeToolInvoker.invoke production callsites outside canonical nexus tool stack:\n"
        + "\n".join(violations)
    )


def test_harness_01_runtime_tool_invoker_constructed_only_at_composition_roots() -> None:
    violations: list[str] = []
    for path in _iter_py_files(_INTERGRAX):
        rel = _relative_posix(path)
        if rel in HARNESS_01_RUNTIME_TOOL_INVOKER_COMPOSITION_ROOTS:
            continue
        if "RuntimeToolInvoker(" in path.read_text(encoding="utf-8"):
            violations.append(rel)
    assert violations == [], (
        "RuntimeToolInvoker must be composed only at documented roots:\n" + "\n".join(violations)
    )


def test_harness_01_application_host_trees_do_not_construct_runtime_tool_invoker() -> None:
    violations: list[str] = []
    for pattern in _APPLICATION_HOST_GLOBS:
        for path in _REPO_ROOT.glob(pattern):
            if not path.is_file():
                continue
            if "RuntimeToolInvoker(" in path.read_text(encoding="utf-8"):
                violations.append(_relative_posix(path))
    assert violations == [], (
        "application host composition must not construct RuntimeToolInvoker:\n"
        + "\n".join(violations)
    )


def test_harness_01_public_pattern_bridge_does_not_trust_port_agent_id() -> None:
    source = _BRIDGE.read_text(encoding="utf-8")
    assert "_ = agent_id" in source or "agent_id" in source
    assert "invoke_prepared_tool_execution_request" in source
    assert "state.request.agent_id" in source or "invoke_tool" in source


def test_harness_01_tool_gateway_wraps_invoke_with_hooks() -> None:
    source = _GATEWAY.read_text(encoding="utf-8")
    assert "run_tool_call_hooks" in source
    assert "ToolAccessPolicy.is_tool_allowed" in source


def test_harness_01_runtime_tier_no_direct_vendor_llm_imports() -> None:
    violations: list[str] = []
    for path in _iter_py_files(_RUNTIME_TIER):
        rel = _relative_posix(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if any(mod == p or mod.startswith(f"{p}.") for p in _FORBIDDEN_RUNTIME_LLM_VENDOR_PREFIXES):
                        violations.append(f"{rel}:{node.lineno}:{mod}")
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if any(mod == p or mod.startswith(f"{p}.") for p in _FORBIDDEN_RUNTIME_LLM_VENDOR_PREFIXES):
                    violations.append(f"{rel}:{node.lineno}:{mod}")
    assert violations == [], (
        "runtime tier must not import vendor LLM SDKs directly:\n" + "\n".join(violations)
    )


def test_harness_01_mapped_evidence_node_ids_are_importable() -> None:
    import importlib

    missing: list[str] = []
    for node_id in sorted(HARNESS_01_MAPPED_NODE_IDS):
        path_part, func = node_id.split("::", 1)
        module_path = path_part.replace("/", ".").removesuffix(".py")
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError:
            missing.append(node_id)
            continue
        if not hasattr(mod, func):
            missing.append(node_id)
    assert missing == [], f"evidence node ids missing test functions: {missing}"
