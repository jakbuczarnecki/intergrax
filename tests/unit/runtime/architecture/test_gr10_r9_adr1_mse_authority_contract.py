# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-ADR1 — canonical MSE authority contract ADR gates and R9 defect inventory."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ADR_PATH = (
    REPO_ROOT
    / "docs/project/technical/adr/entries/2026-09-19/ADR-GR-10-002.md"
)
NEXUS_PORT_PATH = (
    REPO_ROOT
    / "intergrax/runtime/nexus/tools/meaningful_side_effect_authorization_port.py"
)
ORCH_COMPOSITION_PATH = (
    REPO_ROOT
    / "intergrax/runtime/nexus/tools/orchestration_meaningful_side_effect_composition.py"
)
INVOKER_COMPOSITION_PATH = (
    REPO_ROOT
    / "intergrax/runtime/nexus/tools/runtime_tool_invoker_composition.py"
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_gr10_r9_adr1_adr_exists_and_accepted() -> None:
    assert ADR_PATH.is_file()
    adr = _read(ADR_PATH)
    assert "| **Status** | Accepted |" in adr
    assert "GR-10-R9-R1" in adr
    assert "intergrax/contracts" in adr
    assert "MeaningfulSideEffectAuthorizationResult" in adr


def test_gr10_r9_adr1_adr_forbids_object_return_and_synthetic_authority() -> None:
    adr = _read(ADR_PATH)
    assert "Forbidden:** `object`" in adr or "must not return `object`" in adr.lower() or "`object`" in adr
    assert "no synthetic membership" in adr.lower() or "Synthetic `WorkspaceMembership`" in adr
    assert "no default ALLOW" in adr.lower() or "Default `PolicyAction.ALLOW`" in adr
    assert "InMemory" in adr


def test_gr10_r9_adr1_weak_object_return_detected_in_nexus_port() -> None:
    """Inventory: R9 Nexus port still uses unacceptable ``-> object`` until GR-10-R9-R1."""
    source = _read(NEXUS_PORT_PATH)
    tree = ast.parse(source, filename=str(NEXUS_PORT_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "authorize":
            assert node.returns is not None
            ann = ast.unparse(node.returns)
            assert "object" in ann, (
                "expected R9 baseline weak return annotation; missing means port was fixed "
                "without updating ADR1 inventory gate"
            )
            return
    pytest.fail("authorize not found on MeaningfulSideEffectAuthorizationPort")


def test_gr10_r9_adr1_synthetic_authority_detected_in_orchestration_composition() -> None:
    source = _read(ORCH_COMPOSITION_PATH)
    assert "class _PrincipalAuthorityRepository" in source
    assert "PrincipalAuthorityGrant(" in source
    assert "class _PrincipalMembershipRepository" in source
    assert "WorkspaceMembership(" in source


def test_gr10_r9_adr1_default_allow_detected_in_orchestration_composition() -> None:
    source = _read(ORCH_COMPOSITION_PATH)
    assert "PolicyAction.ALLOW" in source
    assert "orchestration.tool_invocation.allow" in source


def test_gr10_r9_adr1_inmemory_production_fallback_detected() -> None:
    source = _read(ORCH_COMPOSITION_PATH)
    for symbol in (
        "InMemoryPrincipalAuthorityRepository",
        "InMemoryWorkspaceMembershipRepository",
        "InMemoryCollaborativePolicyRepository",
        "InMemoryAuthorityDelegationRepository",
    ):
        assert symbol in source


def test_gr10_r9_adr1_fixed_clock_detected_in_orchestration_composition() -> None:
    source = _read(ORCH_COMPOSITION_PATH)
    assert re.search(r"datetime\s*\(\s*2026\s*,", source) is not None
    assert "_COMPOSITION_CLOCK" in source


def test_gr10_r9_adr1_production_invoker_still_auto_wires_default_boundary() -> None:
    """Inventory: production_mode auto-default is an ADR-rejected pattern until R9-R1."""
    source = _read(INVOKER_COMPOSITION_PATH)
    assert "build_default_orchestration_meaningful_side_effect_authorization_boundary" in source


def test_gr10_r9_adr1_matrix_honesty_orchestration_mse_partial() -> None:
    row = next(r for r in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if r.capability == "MSE")
    assert row.coverage is Gr10CoverageStatus.PARTIAL
    assert gr10_matrix_orchestration_status("MSE") is Gr10CoverageStatus.PARTIAL


def test_gr10_r9_adr1_boundary_authorize_is_typed_at_runtime_impl() -> None:
    from intergrax.runtime.policy.meaningful_side_effect_authorization import (
        MeaningfulSideEffectAuthorizationBoundary,
        MeaningfulSideEffectAuthorizationResult,
    )

    sig = inspect.signature(MeaningfulSideEffectAuthorizationBoundary.authorize)
    assert sig.return_annotation is not object
    assert "MeaningfulSideEffectAuthorizationResult" in str(sig.return_annotation)
