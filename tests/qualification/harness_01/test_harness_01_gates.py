# © Artur Czarnecki. All rights reserved.

"""HARNESS-01 — canonical execution path and zero-bypass qualification gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from tests.qualification.harness_01.catalog import (
    HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES,
    HARNESS_01_CANONICAL_MINIMUM_PROOF_KINDS,
    HARNESS_01_EXECUTION_MATRIX,
    HARNESS_01_FORBIDDEN_DIRECT_VENDOR_SDK_PREFIXES,
    HARNESS_01_INDEPENDENT_ZERO_BYPASS_GATE_TEST_NAMES,
    HARNESS_01_MAPPED_NODE_IDS,
    HARNESS_01_REQUIRED_FLOW_IDS,
    HARNESS_01_RUNTIME_TOOL_INVOKER_COMPOSITION_ROOTS,
    HARNESS_01_RUNTIME_TOOL_INVOKER_REFERENCE_ALLOWLIST,
    HARNESS_01_VENDOR_SDK_ALLOWED_REL_PREFIXES,
    HARNESS_01_ZERO_BYPASS_FINDINGS,
)
from tests.qualification.harness_01.evidence_manifest import HARNESS_01_EVIDENCE_EXECUTION_BATCHES as _BATCHES
from tests.qualification.harness_01.invoker_callsite_detector import (
    collect_governed_invoker_callsites,
    file_references_runtime_tool_invoker,
    is_governed_runtime_tool_invoker_invoke_call,
)
from tests.qualification.harness_01.production_scope import (
    iter_application_host_py_files,
    iter_production_agent_py_files,
    iter_production_intergrax_py_files,
    relative_posix,
    repo_root,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = repo_root()
_INTERGRAX = _REPO_ROOT / "intergrax"
_BRIDGE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "public_tool_invocation_pattern_bridge.py"
)
_TOOL_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "tool_loop.py"
_COMP_SESSION = _REPO_ROOT / "intergrax" / "agents" / "persistence" / "compensation_tool_invoke_session.py"
_DECL_PORT = _REPO_ROOT / "intergrax" / "contracts" / "execution_bound_declarative_tool_invocation.py"


def _collect_production_governed_invoker_callsites() -> dict[str, list[int]]:
    hits: dict[str, list[int]] = {}
    for path in iter_production_intergrax_py_files():
        rel = relative_posix(path)
        try:
            source = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        try:
            callsites = collect_governed_invoker_callsites(source, filename=rel)
        except SyntaxError:
            continue
        if callsites:
            hits[rel] = [site.line for site in callsites]
    return hits


def _collect_runtime_tool_invoker_reference_files() -> set[str]:
    refs: set[str] = set()
    for path in iter_production_intergrax_py_files():
        rel = relative_posix(path)
        source = path.read_text(encoding="utf-8-sig")
        if file_references_runtime_tool_invoker(source):
            refs.add(rel)
    agents_root = _INTERGRAX / "agents"
    for path in iter_production_agent_py_files():
        rel = relative_posix(path)
        source = path.read_text(encoding="utf-8-sig")
        if file_references_runtime_tool_invoker(source):
            refs.add(rel)
    return refs


def _nexus_imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("intergrax.runtime.nexus"):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod == "intergrax.runtime.nexus" or mod.startswith("intergrax.runtime.nexus."):
                    names.add(alias.asname or mod.split(".")[-1])
    return names


def _collect_private_nexus_member_access_violations() -> list[str]:
    violations: list[str] = []
    scan_roots = list(iter_production_intergrax_py_files())
    scan_roots.extend(iter_production_agent_py_files())
    scan_roots.extend(iter_application_host_py_files())
    for path in scan_roots:
        rel = relative_posix(path)
        if rel.startswith("intergrax/runtime/nexus/"):
            continue
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError:
            continue
        nexus_names = _nexus_imported_names(tree)
        if not nexus_names:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not node.attr.startswith("_") or node.attr.startswith("__"):
                continue
            if isinstance(node.value, ast.Name) and node.value.id in nexus_names:
                violations.append(f"{rel}:{node.lineno}:{node.value.id}{node.attr}")
    return violations


def _module_imports_forbidden_vendor(mod: str) -> bool:
    return any(mod == prefix or mod.startswith(f"{prefix}.") for prefix in HARNESS_01_FORBIDDEN_DIRECT_VENDOR_SDK_PREFIXES)


def _collect_forbidden_vendor_import_violations(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        rel = relative_posix(path)
        if any(rel.startswith(prefix) for prefix in HARNESS_01_VENDOR_SDK_ALLOWED_REL_PREFIXES):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _module_imports_forbidden_vendor(alias.name):
                        violations.append(f"{rel}:{node.lineno}:{alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module and _module_imports_forbidden_vendor(node.module):
                violations.append(f"{rel}:{node.lineno}:{node.module}")
    return violations


def test_harness_01_matrix_covers_required_flow_id_inventory() -> None:
    actual = {row.flow_id for row in HARNESS_01_EXECUTION_MATRIX}
    assert actual == HARNESS_01_REQUIRED_FLOW_IDS


def test_harness_01_matrix_flow_ids_are_unique() -> None:
    flow_ids = [row.flow_id for row in HARNESS_01_EXECUTION_MATRIX]
    assert len(flow_ids) == len(set(flow_ids))


def test_harness_01_matrix_rows_classify_bypass_status() -> None:
    allowed = {"CANONICAL", "AUTHORIZED_INTERNAL", "BYPASS", "NOT_APPLICABLE"}
    for row in HARNESS_01_EXECUTION_MATRIX:
        assert row.bypass_status in allowed
        if row.bypass_status == "CANONICAL":
            assert row.proof, f"canonical flow {row.flow_id!r} must cite proof node ids"
            kinds = {kind for ref in row.proof for kind in ref.kinds}
            assert kinds & HARNESS_01_CANONICAL_MINIMUM_PROOF_KINDS, (
                f"canonical flow {row.flow_id!r} needs boundary/consumption proof, got {kinds!r}"
            )


def test_harness_01_zero_bypass_findings_index_reports_no_blocker_rows() -> None:
    """Reporting layer only — PASS must not depend on this alone."""
    severities = {row.severity for row in HARNESS_01_ZERO_BYPASS_FINDINGS}
    assert "BLOCKER" not in severities


def test_harness_01_independent_zero_bypass_gate_registry_is_complete() -> None:
    mod = globals()
    for name in sorted(HARNESS_01_INDEPENDENT_ZERO_BYPASS_GATE_TEST_NAMES):
        assert name in mod, f"missing independent gate test {name!r}"


def test_harness_01_runtime_tool_invoker_callsites_are_authorized_internal() -> None:
    hits = _collect_production_governed_invoker_callsites()
    violations: list[str] = []
    for rel, lines in sorted(hits.items()):
        if rel in HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES:
            continue
        violations.append(f"{rel}:{lines}")
    assert violations == [], (
        "RuntimeToolInvoker.invoke production callsites outside canonical nexus tool stack:\n"
        + "\n".join(violations)
    )


def test_harness_01_runtime_tool_invoker_reference_files_are_classified() -> None:
    refs = _collect_runtime_tool_invoker_reference_files()
    unclassified = sorted(refs - HARNESS_01_RUNTIME_TOOL_INVOKER_REFERENCE_ALLOWLIST)
    assert unclassified == [], (
        "Production files referencing RuntimeToolInvoker must be explicitly classified:\n"
        + "\n".join(unclassified)
    )


def test_harness_01_runtime_tool_invoker_constructed_only_at_composition_roots() -> None:
    violations: list[str] = []
    for path in iter_production_intergrax_py_files():
        rel = relative_posix(path)
        if rel in HARNESS_01_RUNTIME_TOOL_INVOKER_COMPOSITION_ROOTS:
            continue
        if "RuntimeToolInvoker(" in path.read_text(encoding="utf-8-sig"):
            violations.append(rel)
    assert violations == [], (
        "RuntimeToolInvoker must be composed only at documented roots:\n" + "\n".join(violations)
    )


def test_harness_01_application_host_trees_do_not_construct_runtime_tool_invoker() -> None:
    violations: list[str] = []
    for path in iter_application_host_py_files():
        if "RuntimeToolInvoker(" in path.read_text(encoding="utf-8-sig"):
            violations.append(relative_posix(path))
    assert violations == [], (
        "application host composition must not construct RuntimeToolInvoker:\n"
        + "\n".join(violations)
    )


def test_harness_01_public_pattern_bridge_ignores_port_agent_id_statically() -> None:
    source = _BRIDGE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_posix(_BRIDGE))
    invoke_tool_methods: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_RuntimeToolInvocationInvokerPort":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "invoke_tool":
                    invoke_tool_methods.append(item)
    assert invoke_tool_methods, "expected _RuntimeToolInvocationInvokerPort.invoke_tool"
    body = invoke_tool_methods[0].body
    ignores_port_agent = any(
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "_"
        and isinstance(stmt.value, ast.Name)
        and stmt.value.id == "agent_id"
        for stmt in body
    )
    assert ignores_port_agent, "bridge must not authorize port-supplied agent_id"
    assert "invoke_prepared_tool_execution_request" in source


def test_harness_01_tool_budget_record_precedes_invoker_invoke() -> None:
    source = _TOOL_LOOP.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_posix(_TOOL_LOOP))
    finish_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_finish_canonical_tool_invocation"
    ]
    assert finish_defs, "_finish_canonical_tool_invocation must exist"
    body = finish_defs[0].body
    record_line: int | None = None
    invoke_line: int | None = None
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Name) and func.id == "record_tool_call_and_enforce":
                record_line = stmt.lineno
        if isinstance(stmt, ast.Try):
            for inner in ast.walk(stmt):
                if isinstance(inner, ast.Call) and is_governed_runtime_tool_invoker_invoke_call(inner):
                    invoke_line = inner.lineno
    assert record_line is not None, "record_tool_call_and_enforce must run in finish helper"
    assert invoke_line is not None, "invoker.invoke must run in finish helper"
    assert record_line < invoke_line, "budget record must precede physical invoker.invoke"


def test_harness_01_declarative_compensation_is_separate_contract_plane() -> None:
    session_source = _COMP_SESSION.read_text(encoding="utf-8")
    assert "ExecutionBoundDeclarativeToolInvoker" in session_source
    assert "RuntimeToolInvoker" not in session_source
    port_source = _DECL_PORT.read_text(encoding="utf-8")
    assert "class ExecutionBoundDeclarativeToolInvoker" in port_source
    hits = collect_governed_invoker_callsites(session_source, filename=relative_posix(_COMP_SESSION))
    assert hits == [], "declarative compensation must not use governed RuntimeToolInvoker.invoke shape"


def test_harness_01_cross_layer_private_nexus_member_access() -> None:
    violations = _collect_private_nexus_member_access_violations()
    assert violations == [], (
        "Higher layers must not access private Nexus members on imported Nexus objects:\n"
        + "\n".join(violations)
    )


def test_harness_01_runtime_agents_and_hosts_no_direct_forbidden_vendor_sdk() -> None:
    runtime_paths = [
        p
        for p in iter_production_intergrax_py_files()
        if relative_posix(p).startswith("intergrax/runtime/")
    ]
    agent_paths = iter_production_agent_py_files()
    host_paths = iter_application_host_py_files()
    violations = _collect_forbidden_vendor_import_violations(runtime_paths + agent_paths + host_paths)
    assert violations == [], (
        "runtime/agents/application host tiers must not import vendor LLM SDKs directly:\n"
        + "\n".join(violations)
    )


def test_harness_01_runtime_tier_no_direct_vendor_llm_imports() -> None:
    runtime_root = _REPO_ROOT / "intergrax" / "runtime"
    violations = _collect_forbidden_vendor_import_violations(
        [p for p in iter_production_intergrax_py_files() if p.is_relative_to(runtime_root)]
    )
    assert violations == [], (
        "runtime tier must not import vendor LLM SDKs directly:\n" + "\n".join(violations)
    )


def test_harness_01_mapped_evidence_references_exist() -> None:
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
    assert missing == [], f"mapped evidence references missing test functions: {missing}"


def test_harness_01_evidence_manifest_batches_are_unique_and_present() -> None:
    labels = [batch.label for batch in _BATCHES]
    assert len(labels) == len(set(labels))
    for batch in _BATCHES:
        target = _REPO_ROOT / batch.pytest_target
        assert target.exists(), f"evidence batch target missing: {batch.pytest_target}"


def test_harness_01_governed_invoker_detector_synthetic_receiver_forms() -> None:
    cases = (
        "invoker.invoke(state=state, request=req)",
        "runtime_invoker.invoke(state=state, request=req)",
        "self._invoker.invoke(state=state, request=req)",
        "ctx.tool_invoker.invoke(state=state, request=req)",
        "foo.bar.baz.invoke(state=state, request=req)",
    )
    for snippet in cases:
        wrapped = f"def _run():\n    {snippet}\n"
        hits = collect_governed_invoker_callsites(wrapped)
        assert hits, f"detector must recognize governed invoke for: {snippet!r}"


def test_harness_01_governed_invoker_detector_ignores_declarative_port_shape() -> None:
    snippet = (
        "async def _run():\n"
        "    return await self._invoker.invoke(tool_id='t', args={}, idempotency_key='k')\n"
    )
    hits = collect_governed_invoker_callsites(snippet)
    assert hits == []


def test_harness_01_synthetic_unauthorized_callsite_would_fail_allowlist() -> None:
    synthetic = (
        "class Bypass:\n"
        "    def run(self, state, req, inv):\n"
        "        return self._invoker.invoke(state=state, request=req)\n"
    )
    hits = collect_governed_invoker_callsites(synthetic)
    assert hits
    rel = "intergrax/foo/bypass.py"
    assert rel not in HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES
