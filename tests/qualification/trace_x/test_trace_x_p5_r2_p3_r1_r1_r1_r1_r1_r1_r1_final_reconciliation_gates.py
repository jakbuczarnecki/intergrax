# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1 target compatibility & intent identity final reconciliation gates."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_START_HEAD = "5f348257e7ff506f57a2b8f381c131ee0e62599f"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_R1_R1_TARGET_COMPATIBILITY_AND_INTENT_IDENTITY_FINAL_RECONCILIATION.md"
)
_PARENT_LOCK = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_R1_MARKETPLACE_BINDING_TARGET_INTENT_RECONCILIATION_LOCK.md"
)
_ROADMAP = _REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
_TARGET_CONTRACT = (
    _REPO_ROOT / "intergrax/contracts/capability_qualification/qualified_capability_binding.py"
)
_INTENT_REPO = _REPO_ROOT / "intergrax/tools/qualified_marketplace_tool_execution_intent_repository.py"
_TOOLS_ROOT = _REPO_ROOT / "intergrax/tools"
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"


def _production_py_files() -> list[Path]:
    roots = (_INTERGRAX_ROOT, _REPO_ROOT / "agents", _REPO_ROOT / "applications")
    return [path for root in roots for path in root.rglob("*.py")]


def test_txp5r2p3r1r1r1r1r1r1r1r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r1r1r1r1r1r1r1r1_q02_final_reconciliation_lock_ready_for_audit() -> None:
    assert _LOCK_DOC.is_file()
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in text
    assert "R2-P3-EXECUTION-TARGET-COMPATIBILITY-WITHOUT-EVIDENCE-21" in text
    assert "R2-P3-INTENT-CAPABILITY-IDENTITY-DUPLICATION-22" in text
    assert "NEW SEMANTIC MECHANISM" in text and "0" in text
    assert "FRZ-TRC-11" in text and "OPEN" in text
    assert "derive_marketplace_configured_tool_execution_target_reference" in text
    assert "registry.resolve(target.execution_handler_id)" in text


def test_txp5r2p3r1r1r1r1r1r1r1r1_q03_parent_lock_superseded_for_21_22() -> None:
    parent = _PARENT_LOCK.read_text(encoding="utf-8")
    assert "BLOCKED / SUPERSEDED BY CHILD" in parent
    assert "TARGET_COMPATIBILITY_AND_INTENT_IDENTITY_FINAL_RECONCILIATION" in parent


def test_txp5r2p3r1r1r1r1r1r1r1r1_q04_no_target_persistence_repository() -> None:
    hits: list[str] = []
    for path in _production_py_files():
        if path.name.endswith("_repository.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "QualifiedCapabilityExecutionTarget" in text:
                hits.append(str(path.relative_to(_REPO_ROOT)))
    assert hits == []


def test_txp5r2p3r1r1r1r1r1r1r1r1_q05_target_schema_only_in_binding_contract() -> None:
    schema_hits: list[str] = []
    for path in _production_py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "qualified_capability_execution_target" in text:
            schema_hits.append(str(path.relative_to(_REPO_ROOT)))
    assert {p.replace("\\", "/") for p in schema_hits} == {
        "intergrax/contracts/capability_qualification/qualified_capability_binding.py"
    }


def test_txp5r2p3r1r1r1r1r1r1r1r1_q06_target_is_runtime_contract_field() -> None:
    tree = ast.parse(_TARGET_CONTRACT.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "QualifiedCapabilityExecutionTarget"
        for node in tree.body
    )
    dispatch = _REPO_ROOT / "intergrax/contracts/execution/qualified_capability_execution_dispatch.py"
    assert "QualifiedCapabilityExecutionTarget" in dispatch.read_text(encoding="utf-8")


def test_txp5r2p3r1r1r1r1r1r1r1r1_q07_no_target_v1_v2_runtime_adapter_in_production() -> None:
    forbidden_fragments = (
        "QualifiedCapabilityExecutionTargetMigration",
        "ExecutionTargetCompatibility",
    )
    allowed_v2_contract = (
        _INTERGRAX_ROOT / "contracts/capability_qualification/qualified_capability_binding.py"
    )
    scoped_roots = (
        _INTERGRAX_ROOT / "contracts/execution",
        _INTERGRAX_ROOT / "runtime/execution",
        _INTERGRAX_ROOT / "tools",
        _INTERGRAX_ROOT / "autonomous_work",
    )
    hits: list[str] = []
    binding = allowed_v2_contract.read_text(encoding="utf-8")
    assert "qualified_capability_execution_target.v2" in binding
    for root in scoped_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "QualifiedCapabilityExecutionTarget" not in text:
                continue
            for frag in forbidden_fragments:
                if frag in text:
                    hits.append(f"{path.relative_to(_REPO_ROOT)}:{frag}")
    assert hits == []


def test_txp5r2p3r1r1r1r1r1r1r1r1_q08_lock_forbids_target_runtime_compatibility_without_evidence() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "runtime handoff only" in lock.lower() or "Runtime handoff only" in lock
    assert "no runtime target compatibility" in lock.lower() or "NO target compatibility runtime" in lock
    assert "STOP — ARCHITECTURE DECISION REQUIRED" in lock


def test_txp5r2p3r1r1r1r1r1r1r1r1_q09_capability_identity_once_on_common_intent() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "MarketplaceToolExecutionIntent.capability_identity" in lock
    assert "exactly once" in lock.lower() or "Exactly-one" in lock or "exactly once" in lock


def test_txp5r2p3r1r1r1r1r1r1r1r1_q10_configured_provenance_must_not_own_capability_identity() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "ConfiguredMarketplaceToolExecutionProvenance" in lock
    assert "DUPLICATE / ELIMINATE" in lock or "Must not contain" in lock
    assert "provenance.capability_identity" in lock or "duplicate" in lock.lower()


def test_txp5r2p3r1r1r1r1r1r1r1r1_q11_deterministic_opaque_configured_target_reference() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "binding_operation_id" in lock
    assert "Opaque" in lock or "opaque" in lock
    assert "non-semantic" in lock.lower() or "non-semantic" in lock


def test_txp5r2p3r1r1r1r1r1r1r1r1_q12_intent_repository_owns_durable_compatibility() -> None:
    repo_src = _INTENT_REPO.read_text(encoding="utf-8")
    assert "ConditionalDocumentStore" in repo_src
    assert "model_dump" in repo_src
    assert "model_validate" in repo_src
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository" in lock
    assert "runtime schema compatibility branch" in lock.lower()


def test_txp5r2p3r1r1r1r1r1r1r1r1_q13_single_intent_repository_implementation() -> None:
    assert _INTENT_REPO.is_file()
    assert not (_TOOLS_ROOT / "configured_marketplace_tool_execution_intent_repository.py").is_file()
    assert not (_TOOLS_ROOT / "marketplace_tool_execution_intent_repository.py").is_file()


def test_txp5r2p3r1r1r1r1r1r1r1r1_q14_roadmap_child_ready_for_audit() -> None:
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    assert "TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1" in roadmap
    assert "TARGET_COMPATIBILITY_AND_INTENT_IDENTITY_FINAL_RECONCILIATION" in roadmap
    assert "TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1" in roadmap
    assert "BLOCKED / SUPERSEDED BY CHILD" in roadmap


def test_txp5r2p3r1r1r1r1r1r1r1r1_q15_frz_trc_11_remains_open() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "FRZ-TRC-11" in lock
    assert "OPEN" in lock
