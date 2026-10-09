# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1 binding/target/intent reconciliation architecture gates."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_START_HEAD = "dc8795043e08ff57af51fac7b88177a31d8f9194"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_R1_MARKETPLACE_BINDING_TARGET_INTENT_RECONCILIATION_LOCK.md"
)
_PRIOR_LOCK = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_CANONICAL_TOOL_EXECUTION_CONVERGENCE_LOCK.md"
)
_ROADMAP = _REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
_TARGET_CONTRACT = (
    _REPO_ROOT / "intergrax/contracts/capability_qualification/qualified_capability_binding.py"
)
_HANDLER_REGISTRY = (
    _REPO_ROOT / "intergrax/runtime/execution/qualified_capability_execution_handlers.py"
)
_MARKETPLACE_HANDLER = (
    _REPO_ROOT / "intergrax/tools/marketplace_qualified_capability_execution_handler.py"
)
_MARKETPLACE_BINDING = (
    _REPO_ROOT / "intergrax/tools/marketplace_qualified_capability_binding_provider.py"
)
_IDENTITY_KEY = _REPO_ROOT / "intergrax/contracts/capability_catalog/identity_key.py"
_INTENT_CONTRACT = (
    _REPO_ROOT / "intergrax/contracts/tools/qualified_marketplace_tool_execution_intent.py"
)
_INTENT_REPO = _REPO_ROOT / "intergrax/tools/qualified_marketplace_tool_execution_intent_repository.py"
_TOOLS_ROOT = _REPO_ROOT / "intergrax/tools"
_RUNTIME_DELEGATE = (
    _REPO_ROOT / "intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py"
)


def test_txp5r2p3r1r1r1r1r1r1r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r1r1r1r1r1r1r1_q02_reconciliation_lock_ready_for_audit() -> None:
    assert _LOCK_DOC.is_file()
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "BLOCKED / SUPERSEDED BY CHILD" in text
    assert "TARGET_COMPATIBILITY_AND_INTENT_IDENTITY_FINAL_RECONCILIATION" in text
    assert "R2-P3-BINDING-PROVIDER-IDENTITY-CONFLATION-18" in text
    assert "R2-P3-EXECUTION-TARGET-STRING-CONTRACT-19" in text
    assert "R2-P3-TOOL-INTENT-PROVENANCE-CONFLATION-20" in text
    assert "execution_handler_id" in text
    assert "marketplace.tool.execution.v1" in text
    assert "FRZ-TRC-11" in text and "OPEN" in text
    assert "NEW SEMANTIC MECHANISM = 0" in text


def test_txp5r2p3r1r1r1r1r1r1r1_q03_prior_tool_lock_superseded() -> None:
    prior = _PRIOR_LOCK.read_text(encoding="utf-8")
    assert "BLOCKED / SUPERSEDED BY CHILD" in prior
    assert "MARKETPLACE_BINDING_TARGET_INTENT_RECONCILIATION_LOCK" in prior


def test_txp5r2p3r1r1r1r1r1r1r1_q04_registry_resolves_by_binding_provider_id() -> None:
    tree = ast.parse(_HANDLER_REGISTRY.read_text(encoding="utf-8"))
    registry_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "QualifiedCapabilityExecutionBindingHandlerRegistry"
    )
    resolve_fn = next(
        inner
        for inner in registry_cls.body
        if isinstance(inner, ast.FunctionDef) and inner.name == "resolve"
    )
    assert len(resolve_fn.args.args) >= 2
    assert resolve_fn.args.args[1].arg == "binding_provider_id"
    source = _RUNTIME_DELEGATE.read_text(encoding="utf-8")
    assert "execution_target.binding_provider_id" in source


def test_txp5r2p3r1r1r1r1r1r1r1_q05_handler_protocol_binding_provider_id_property() -> None:
    tree = ast.parse(_HANDLER_REGISTRY.read_text(encoding="utf-8"))
    protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "QualifiedCapabilityExecutionBindingHandler"
    )
    props = [
        node
        for node in protocol.body
        if isinstance(node, ast.FunctionDef) and node.name == "binding_provider_id"
    ]
    assert props
    handler_src = _MARKETPLACE_HANDLER.read_text(encoding="utf-8")
    assert "def binding_provider_id" in handler_src


def test_txp5r2p3r1r1r1r1r1r1r1_q06_target_v1_has_no_execution_handler_id() -> None:
    tree = ast.parse(_TARGET_CONTRACT.read_text(encoding="utf-8"))
    target_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QualifiedCapabilityExecutionTarget"
    )
    field_names = {
        stmt.target.id
        for stmt in target_cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    assert "binding_provider_id" in field_names
    assert "execution_handler_id" not in field_names


def test_txp5r2p3r1r1r1r1r1r1r1_q07_qualified_binding_provider_id_is_uca_constant() -> None:
    binding_src = _MARKETPLACE_BINDING.read_text(encoding="utf-8")
    assert "MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID" in binding_src
    assert '"marketplace.tool.qualified_binding.v1"' in binding_src
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "configured_capability_binding" in lock
    assert "must not" in lock.lower() or "Forbidden" in lock or "FORBIDDEN" in lock


def test_txp5r2p3r1r1r1r1r1r1r1_q08_capability_identity_key_is_typed_class() -> None:
    tree = ast.parse(_IDENTITY_KEY.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "CapabilityIdentityKey"
        for node in tree.body
    )


def test_txp5r2p3r1r1r1r1r1r1r1_q09_no_catalog_tool_capability_parser_in_production() -> None:
    hits: list[Path] = []
    for root in (_REPO_ROOT / "intergrax", _REPO_ROOT / "agents", _REPO_ROOT / "applications"):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "catalog-tool-capability" in text:
                hits.append(path)
    assert hits == []


def test_txp5r2p3r1r1r1r1r1r1r1_q10_intent_v1_requires_handoff_and_resume() -> None:
    source = _INTENT_CONTRACT.read_text(encoding="utf-8")
    assert "handoff_id: str" in source
    assert "resume_operation_id: str" in source


def test_txp5r2p3r1r1r1r1r1r1r1_q11_single_document_store_intent_repository() -> None:
    assert _INTENT_REPO.is_file()
    repo_src = _INTENT_REPO.read_text(encoding="utf-8")
    assert "ConditionalDocumentStore" in repo_src
    assert "intergrax.qualified_marketplace_tool_execution_intent.v1" in repo_src
    assert not (_TOOLS_ROOT / "configured_marketplace_tool_execution_intent_repository.py").is_file()


def test_txp5r2p3r1r1r1r1r1r1r1_q12_lock_forbids_nullable_provenance_soup() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "nullable" in lock.lower() or "optional fields" in lock.lower()
    assert "UcaMarketplaceToolExecutionProvenance" in lock
    assert "ConfiguredMarketplaceToolExecutionProvenance" in lock


def test_txp5r2p3r1r1r1r1r1r1r1_q13_single_handler_registry_class() -> None:
    tree = ast.parse(_HANDLER_REGISTRY.read_text(encoding="utf-8"))
    registries = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and "HandlerRegistry" in node.name
        and node.name.endswith("Registry")
    ]
    assert registries == ["QualifiedCapabilityExecutionBindingHandlerRegistry"]


def test_txp5r2p3r1r1r1r1r1r1r1_q14_roadmap_current_child_ready_for_audit() -> None:
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    assert "TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1" in roadmap
    assert "MARKETPLACE_BINDING_TARGET_INTENT_RECONCILIATION_LOCK" in roadmap
    assert "TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1" in roadmap
    assert "TARGET_COMPATIBILITY_AND_INTENT_IDENTITY_FINAL_RECONCILIATION" in roadmap
    assert "TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1" in roadmap
    assert "BLOCKED / SUPERSEDED BY CHILD" in roadmap


def test_txp5r2p3r1r1r1r1r1r1r1_q15_frz_trc_11_and_dup_x_unchanged() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "FRZ-TRC-11" in lock
    assert "OPEN" in lock
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    assert roadmap.count("### 3.0.3 `DUP-X`") == 1
    assert "**`DUP-X` = CLOSED**" in roadmap
