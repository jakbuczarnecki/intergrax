# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1 Tool execution convergence architecture gates (docs-only)."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_START_HEAD = "79365c021c4637d13edb0e90fe4f65cd14023b88"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_CANONICAL_TOOL_EXECUTION_CONVERGENCE_LOCK.md"
)
_PRIOR_LOCK = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R1_R1_R1_R1_EXISTING_MECHANISM_REUSE_AND_CONVERGENCE_LOCK.md"
)
_ROADMAP = _REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
_IDENTITY_KEY = _REPO_ROOT / "intergrax/contracts/capability_catalog/identity_key.py"
_RELEASE_IDENTITY = _REPO_ROOT / "intergrax/contracts/capability_catalog/release_identity.py"
_KNOWN_REALIZATION = _REPO_ROOT / "intergrax/tools/known_capability_realization.py"
_DYNAMIC_ACQUISITION = _REPO_ROOT / "intergrax/tools/dynamic_acquisition.py"
_REGISTRY_READ = _REPO_ROOT / "intergrax/tools/registry/read.py"
_HANDLER = _REPO_ROOT / "intergrax/tools/marketplace_qualified_capability_execution_handler.py"
_INTENT_CONTRACT = (
    _REPO_ROOT
    / "intergrax/contracts/tools/qualified_marketplace_tool_execution_intent.py"
)
_TOOLS_ROOT = _REPO_ROOT / "intergrax/tools"


def test_txp5r2p3r1r1r1r1r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r1r1r1r1r1_q02_tool_convergence_lock_ready_for_audit() -> None:
    assert _LOCK_DOC.is_file()
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in text
    assert "R2-P3-CAPABILITY-IDENTITY-TO-TOOL-TARGET-RESOLUTION-16" in text
    assert "R2-P3-MARKETPLACE-EXECUTION-LINEAGE-CONVERGENCE-17" in text
    assert "FRZ-TRC-11" in text and "OPEN" in text
    assert "NEW SEMANTIC MECHANISM = 0" in text or "new semantic mechanisms:** **0**" in text


def test_txp5r2p3r1r1r1r1r1_q03_prior_lock_points_to_tool_convergence_child() -> None:
    prior = _PRIOR_LOCK.read_text(encoding="utf-8")
    assert "CANONICAL_TOOL_EXECUTION_CONVERGENCE_LOCK" in prior


def test_txp5r2p3r1r1r1r1r1_q04_capability_identity_key_has_no_tenant_id_field() -> None:
    tree = ast.parse(_IDENTITY_KEY.read_text(encoding="utf-8"))
    class_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CapabilityIdentityKey"
    )
    field_names = {
        target.id
        for stmt in class_def.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        for target in [stmt.target]
    }
    assert "tenant_id" not in field_names


def test_txp5r2p3r1r1r1r1r1_q05_capability_release_identity_exists() -> None:
    source = _RELEASE_IDENTITY.read_text(encoding="utf-8")
    assert "class CapabilityReleaseIdentity" in source


def test_txp5r2p3r1r1r1r1r1_q06_tool_known_capability_realization_service_exists() -> None:
    source = _KNOWN_REALIZATION.read_text(encoding="utf-8")
    assert "class ToolKnownCapabilityRealizationService" in source
    assert "ToolPackageResolutionForIdentityPort" in source


def test_txp5r2p3r1r1r1r1r1_q07_tool_host_activation_port_exists() -> None:
    source = _DYNAMIC_ACQUISITION.read_text(encoding="utf-8")
    assert "class ToolHostActivationPort" in source


def test_txp5r2p3r1r1r1r1r1_q08_tool_registry_read_is_activation_read_surface() -> None:
    source = _REGISTRY_READ.read_text(encoding="utf-8")
    assert "class ToolRegistryRead" in source
    assert "activation_metadata" in source
    assert "not lifecycle mutation" in source.lower() or "read-only" in source.lower()


def test_txp5r2p3r1r1r1r1r1_q09_qualified_handler_uses_handoff_and_stage() -> None:
    source = _HANDLER.read_text(encoding="utf-8")
    assert "handoff_id" in source
    assert "MarketplaceQualifiedToolStage" in source or "stage_repository" in source
    assert "parse_marketplace_qualified_tool_execution_target_reference" in source


def test_txp5r2p3r1r1r1r1r1_q10_qualified_intent_requires_handoff_id() -> None:
    source = _INTENT_CONTRACT.read_text(encoding="utf-8")
    assert "handoff_id: str" in source


def test_txp5r2p3r1r1r1r1r1_q11_no_configured_tool_activation_service() -> None:
    forbidden = list(_TOOLS_ROOT.glob("configured_tool_*activation*.py"))
    assert forbidden == []
    assert not (_TOOLS_ROOT / "configured_tool_activation_resolver.py").is_file()


def test_txp5r2p3r1r1r1r1r1_q12_no_configured_intent_repository() -> None:
    forbidden = _REPO_ROOT / "intergrax/tools/configured_marketplace_tool_execution_intent.py"
    assert not forbidden.is_file()
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "ConfiguredMarketplaceToolExecutionIntentRepository" in lock


def test_txp5r2p3r1r1r1r1r1_q13_no_configured_tool_registry() -> None:
    assert not (_TOOLS_ROOT / "configured_tool_registry.py").is_file()
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "ConfiguredToolRegistry" in lock or "configured_tool_registry" in lock.lower()


def test_txp5r2p3r1r1r1r1r1_q14_lock_rejects_duplicate_activation_and_intent_pipelines() -> None:
    lock = _LOCK_DOC.read_text(encoding="utf-8")
    assert "second intent" in lock.lower() or "one" in lock.lower()
    assert "ToolHostActivationPort" in lock
    assert "Forbidden" in lock or "FORBIDDEN" in lock


def test_txp5r2p3r1r1r1r1r1_q15_dup_x_roadmap_singleton() -> None:
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    assert roadmap.count("### 3.0.3 `DUP-X`") == 1
    assert "**`DUP-X` = CLOSED**" in roadmap


def test_txp5r2p3r1r1r1r1r1_q16_roadmap_lists_current_child() -> None:
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    assert "TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1" in roadmap
    assert "CANONICAL_TOOL_EXECUTION_CONVERGENCE_LOCK" in roadmap
