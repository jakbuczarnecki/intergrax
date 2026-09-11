# © Artur Czarnecki. All rights reserved.

"""U5 — final zero-bypass qualification gates (inventory + EP-14 / EP-17)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.runtime.architecture.test_platform_execution_unification_p0_bypass_inventory import (
    _central_inventory_verdict_counts,
    _inventory_doc_text,
    _parse_central_inventory_rows,
    _parse_metrics_verdict_counts,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DECLARATIVE_WIRING = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "declarative_tool_wiring.py"
)
_ACP_SESSION_HOST_WIRING = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "acp_session_host_wiring.py"
)
_HARNESS_HOST_RUNTIME = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "harness_host_runtime.py"
)
_U5_ACP_TENANT_PROOF = (
    _REPO_ROOT / "tests" / "unit" / "applications" / "test_acp_session_host_wiring.py"
)
_CATALOG_INVOKER = (
    _REPO_ROOT / "intergrax" / "agents" / "persistence" / "catalog_declarative_invoker.py"
)
_PERSISTENCE_PACKAGE = _REPO_ROOT / "intergrax" / "agents" / "persistence" / "__init__.py"
_RUNTIME_CONTEXT = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_context.py"
)
_APPROVED_RUNTIME_TOOL_INVOKER_OWNERS = (
    _DECLARATIVE_WIRING,
    _RUNTIME_CONTEXT,
)
_AGENTS_PRODUCTION_ROOT = _REPO_ROOT / "intergrax" / "agents"
_AW_STAGE_LOOP = _REPO_ROOT / "intergrax" / "autonomous_work" / "work_stage_capability_loop.py"
_U5_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_U5_FINAL_ZERO_BYPASS_QUALIFICATION.md"
)
_PRODUCTION_APP_ROOTS = (
    _REPO_ROOT / "intergrax" / "applications",
    _REPO_ROOT / "applications",
)


def _annotation_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_name(node.left)
        right = _annotation_name(node.right)
        if left and right:
            return f"{left}|{right}"
    return None


def _catalog_declarative_tool_invoker_contract(tree: ast.Module) -> tuple[str | None, bool]:
    """Return (tool_invoker annotation name, has_runtime_tool_invoker_isinstance_dispatch)."""
    annotation: str | None = None
    isinstance_dispatch = False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "CatalogDeclarativeToolInvoker":
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "tool_invoker"
                and stmt.annotation is not None
            ):
                annotation = _annotation_name(stmt.annotation)
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not isinstance(func, ast.Name) or func.id != "isinstance":
                continue
            if len(child.args) < 2:
                continue
            type_arg = child.args[1]
            if _annotation_name(type_arg) == "RuntimeToolInvoker":
                isinstance_dispatch = True
    return annotation, isinstance_dispatch


def test_u5_qualification_artifact_present() -> None:
    assert _U5_QUALIFICATION.is_file()


def test_u5_inventory_has_no_production_bypass_ambiguous_or_execution_gaps() -> None:
    text = _inventory_doc_text()
    rows = _parse_central_inventory_rows(text)
    inventory_counts = _central_inventory_verdict_counts(rows)
    metrics_counts = _parse_metrics_verdict_counts(text)

    for label, count in inventory_counts.items():
        assert metrics_counts[label] == count

    assert inventory_counts["BYPASS"] == 0
    assert inventory_counts["AMBIGUOUS"] == 0
    assert inventory_counts["CANONICAL WITH GAP"] == 0

    by_id = {row["id"]: row for row in rows}
    assert by_id["EP-14"]["verdict"] == "CANONICAL"
    assert by_id["EP-17"]["verdict"] == "LEGACY BUT NON-PRODUCTION"


def test_u5_unmanaged_catalog_builder_not_public_package_surface() -> None:
    init_source = _PERSISTENCE_PACKAGE.read_text(encoding="utf-8")
    tree = ast.parse(init_source, filename=str(_PERSISTENCE_PACKAGE))
    exported: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = node.value
                if isinstance(value, (ast.List, ast.Tuple)):
                    for elt in value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exported.add(elt.value)
    assert "build_catalog_declarative_invoker_from_registry" not in exported
    assert "build_catalog_declarative_invoker_from_registry" not in init_source

    catalog_source = _CATALOG_INVOKER.read_text(encoding="utf-8")
    assert "def build_catalog_declarative_invoker_from_registry" not in catalog_source
    assert "RuntimeToolInvoker(" not in catalog_source


def test_u5_runtime_tool_invoker_construction_sites_are_approved_owners() -> None:
    for path in _APPROVED_RUNTIME_TOOL_INVOKER_OWNERS:
        assert path.is_file()
        assert "RuntimeToolInvoker(" in path.read_text(encoding="utf-8-sig")

    agent_violations: list[str] = []
    for path in _AGENTS_PRODUCTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8-sig")
        if "RuntimeToolInvoker(" in source:
            agent_violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert agent_violations == [], (
        "intergrax/agents must not construct RuntimeToolInvoker locally:\n"
        + "\n".join(agent_violations)
    )


def test_u5_ep14_declarative_wiring_forwards_governance_and_production_mode() -> None:
    wiring_source = _DECLARATIVE_WIRING.read_text(encoding="utf-8")
    assert "agent_runtime_governance=agent_runtime_governance" in wiring_source
    assert "build_declarative_invoker_for_application_host" in wiring_source
    assert "production declarative tool invoker requires agent_runtime_governance" in wiring_source

    catalog_source = _CATALOG_INVOKER.read_text(encoding="utf-8")
    assert "production_mode: bool = False" in catalog_source
    assert "production_mode=self.production_mode" in catalog_source

    catalog_tree = ast.parse(catalog_source, filename=str(_CATALOG_INVOKER))
    tool_invoker_annotation, has_concrete_dispatch = _catalog_declarative_tool_invoker_contract(
        catalog_tree,
    )
    assert tool_invoker_annotation == "RuntimeToolInvoker"
    assert not has_concrete_dispatch


def test_u5_ep17_no_production_wiring_for_work_stage_capability_loop() -> None:
    """EP-17 loop is contract-only in production trees; integration tests own bindings."""
    module_suffix = "work_stage_capability_loop"
    for root in _PRODUCTION_APP_ROOTS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8-sig")
            if module_suffix not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.endswith(module_suffix):
                        rel = path.relative_to(_REPO_ROOT).as_posix()
                        raise AssertionError(
                            f"production tree imports work stage loop: {rel} -> {node.module}",
                        )
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.endswith(module_suffix):
                            rel = path.relative_to(_REPO_ROOT).as_posix()
                            raise AssertionError(
                                f"production tree imports work stage loop: {rel}",
                            )

    assert _AW_STAGE_LOOP.is_file()


def test_u5_acp_session_host_preserves_harness_tenant_identity() -> None:
    acp_source = _ACP_SESSION_HOST_WIRING.read_text(encoding="utf-8")
    assert 'tenant_id=""' not in acp_source
    assert "tenant_id=runtime.tenant_id" in acp_source

    harness_source = _HARNESS_HOST_RUNTIME.read_text(encoding="utf-8")
    assert "tenant_id: str" in harness_source
    assert "tenant_id=resolved_tenant_id" in harness_source


def test_u5_acp_tenant_proof_has_no_private_member_access_or_slf001() -> None:
    source = _U5_ACP_TENANT_PROOF.read_text(encoding="utf-8")
    assert "SLF001" not in source
    assert "_agent_runtime_governance" not in source
    assert "_capability_resolver" not in source
    assert "test_build_acp_session_host_from_harness_strict_tenant_governance" in source

    lines = source.splitlines()
    tree = ast.parse(source)
    scoped_names = {
        "_ProbeHandler",
        "_register_tenant_probe_tool",
        "_strict_harness_with_probe",
        "test_build_acp_session_host_from_harness_strict_tenant_governance_allows_matching_tenant",
        "test_build_acp_session_host_from_harness_strict_tenant_governance_denies_wrong_tenant",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.ClassDef):
            continue
        if node.name not in scoped_names:
            continue
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        block = "\n".join(lines[start : end + 1])
        assert "pyright: ignore" not in block, f"{node.name} must not use pyright suppressions"
