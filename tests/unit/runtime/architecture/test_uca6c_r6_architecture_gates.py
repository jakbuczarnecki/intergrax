# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]

SUSPENDED_CONTRACTS = REPO / "intergrax" / "contracts" / "execution" / "suspended_operation"
SUSPENDED_RUNTIME = REPO / "intergrax" / "runtime" / "execution" / "suspended_operation"
L3_HOST = REPO / "intergrax" / "runtime" / "nexus" / "tools" / "continuation_aware_catalog_tool_host.py"
UCA_COMPOSITION = (
    REPO / "intergrax" / "applications" / "_shared" / "uca6c_codecraft_qualified_execution_composition.py"
)


def _collect_annotation_tokens(tree: ast.AST) -> set[str]:
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        if isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                tokens.add(node.value.id)
            if isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Name):
                        tokens.add(elt.id)
    return tokens


def test_execution_suspended_contracts_do_not_import_nexus() -> None:
    for path in SUSPENDED_CONTRACTS.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "runtime.nexus" not in text


def test_suspended_payload_contracts_avoid_semantic_any() -> None:
    banned = {"Any"}
    for file in SUSPENDED_CONTRACTS.glob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        tokens = _collect_annotation_tokens(tree)
        assert "Any" not in tokens, f"{file.name} contains banned Any"
        text = file.read_text(encoding="utf-8")
        for pattern in ("dict[str, Any]", "Mapping[str, Any]"):
            assert pattern not in text, f"{file.name} contains {pattern}"


def test_l3_host_uses_contract_codec_registry_not_default() -> None:
    text = L3_HOST.read_text(encoding="utf-8")
    assert "DefaultSuspendedOperationCodecRegistry" not in text
    assert "InMemorySuspendedExecutionOperationStore" not in text
    assert "getattr" not in text
    assert "setattr" not in text


def test_suspended_runtime_reentry_uses_codec_contract() -> None:
    path = SUSPENDED_RUNTIME / "reentry_coordinator.py"
    text = path.read_text(encoding="utf-8")
    assert "DefaultSuspendedOperationCodecRegistry" not in text


def test_suspended_scope_bans_reflection() -> None:
    paths = list(SUSPENDED_RUNTIME.glob("*.py")) + [L3_HOST]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {
                    "getattr",
                    "setattr",
                    "eval",
                    "exec",
                }, f"{path.name} uses forbidden {node.func.id}"


def test_uca_composition_wires_continuation_aware_dependencies() -> None:
    text = UCA_COMPOSITION.read_text(encoding="utf-8")
    assert "build_execution_bound_catalog_tool_composition" in text
    assert "validate_document_store_for_production_suspended_operations" in text
    assert "runtime.nexus" not in text
    assert "InMemoryDocumentStore" not in text
    assert "wire_execution_engine_continuation_dependencies" not in text
    assert "wire_sandbox_sessions" not in text
    assert "sandbox_session_manager" not in text
    assert "DurableToolInvocationWiringBindingResolver" in text


def test_reentry_coordinator_does_not_synthesize_pending() -> None:
    text = (SUSPENDED_RUNTIME / "reentry_coordinator.py").read_text(encoding="utf-8")
    assert "_pending_from_pause_descriptor" not in text
    assert "DeclarativeHitlPendingApproval(" not in text
    assert "sandbox_session:" not in text
    assert "SandboxSession" not in text
    assert "binding_resolver" in text


def test_reentry_coordinator_requires_durable_binding_resolver_contract() -> None:
    text = (SUSPENDED_RUNTIME / "reentry_coordinator.py").read_text(encoding="utf-8")
    assert "DurableToolInvocationWiringBindingResolver" in text
    assert "FixedSandboxSessionWiringResolver" not in text


def test_aw_does_not_import_nexus() -> None:
    aw_root = REPO / "intergrax" / "autonomous_work"
    for path in aw_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "runtime.nexus" not in text
