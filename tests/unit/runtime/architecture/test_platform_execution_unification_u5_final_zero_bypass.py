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


def test_u5_ep14_declarative_wiring_forwards_governance_and_production_mode() -> None:
    wiring_source = _DECLARATIVE_WIRING.read_text(encoding="utf-8")
    assert "agent_runtime_governance=agent_runtime_governance" in wiring_source
    assert "build_declarative_invoker_for_application_host" in wiring_source
    assert "production declarative tool invoker requires agent_runtime_governance" in wiring_source

    catalog_source = _CATALOG_INVOKER.read_text(encoding="utf-8")
    assert "production_mode: bool = False" in catalog_source
    assert "production_mode=self.production_mode" in catalog_source


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
