# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-R1 — canonical MSE contract migration and production fail-closed gates."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPO_ROOT / "intergrax/contracts/meaningful_side_effect_authorization.py"
NEXUS_PORT_PATH = (
    REPO_ROOT / "intergrax/runtime/nexus/tools/meaningful_side_effect_authorization_port.py"
)
ORCH_COMPOSITION_PATH = (
    REPO_ROOT
    / "intergrax/runtime/governance/orchestration_meaningful_side_effect_composition.py"
)
INVOKER_COMPOSITION_PATH = (
    REPO_ROOT / "intergrax/runtime/nexus/tools/runtime_tool_invoker_composition.py"
)
OLD_ORCH_NEXUS_PATH = (
    REPO_ROOT
    / "intergrax/runtime/nexus/tools/orchestration_meaningful_side_effect_composition.py"
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_gr10_r9_r1_canonical_contract_importable_without_runtime() -> None:
    spec = importlib.util.find_spec("intergrax.contracts.meaningful_side_effect_authorization")
    assert spec is not None
    module = importlib.import_module("intergrax.contracts.meaningful_side_effect_authorization")
    source = Path(module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.runtime")
            assert not node.module.startswith("intergrax.runtime.nexus")


def test_gr10_r9_r1_port_authorize_returns_typed_result() -> None:
    sig = inspect.signature(MeaningfulSideEffectAuthorizationPort.authorize)
    assert "MeaningfulSideEffectAuthorizationResult" in str(sig.return_annotation)
    assert "object" not in str(sig.return_annotation)
    assert sig.return_annotation is not object


def test_gr10_r9_r1_result_frozen_dataclass() -> None:
    fields = MeaningfulSideEffectAuthorizationResult.__dataclass_fields__
    assert MeaningfulSideEffectAuthorizationResult.__dataclass_params__.frozen is True
    assert set(fields) == {
        "permitted",
        "decision",
        "enforcement_result",
        "requires_governed_continuation",
        "governed_continuation_request",
    }


def test_gr10_r9_r1_boundary_structurally_satisfies_port() -> None:
    sig = inspect.signature(MeaningfulSideEffectAuthorizationBoundary.authorize)
    assert "MeaningfulSideEffectAuthorizationResult" in str(sig.return_annotation)


def test_gr10_r9_r1_nexus_port_is_reexport_only() -> None:
    source = _read(NEXUS_PORT_PATH)
    assert "class MeaningfulSideEffectAuthorizationPort" not in source
    assert "intergrax.contracts.meaningful_side_effect_authorization" in source


def test_gr10_r9_r1_no_nexus_owned_orchestration_production_composition() -> None:
    assert not OLD_ORCH_NEXUS_PATH.is_file()


def test_gr10_r9_r1_governance_explicit_composition_no_inmemory() -> None:
    source = _read(ORCH_COMPOSITION_PATH)
    assert "InMemory" not in source
    assert "PrincipalAuthorityGrant(" not in source
    assert "WorkspaceMembership(" not in source
    assert "PolicyAction.ALLOW" not in source
    assert re.search(r"datetime\s*\(\s*20\d{2}\s*,", source) is None


def test_gr10_r9_r1_production_invoker_missing_mse_port_fail_closed() -> None:
    source = _read(INVOKER_COMPOSITION_PATH)
    assert "build_default_orchestration_meaningful_side_effect_authorization_boundary" not in source
    assert "meaningful_side_effect_authorization is required when production_mode=True" in source


def test_gr10_r9_r1_contract_single_owner() -> None:
    contract_source = _read(CONTRACT_PATH)
    assert "class MeaningfulSideEffectAuthorizationPort" in contract_source
    nexus_source = _read(NEXUS_PORT_PATH)
    assert "class MeaningfulSideEffectAuthorizationPort" not in nexus_source


def test_gr10_r9_r1_graph_non_tool_inventory_no_tbd() -> None:
    assert GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY
    for row in GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY:
        assert row.classification.strip()
        assert row.classification != "TBD"


def test_gr10_r9_r1_matrix_mse_partial_honest_after_authority_migration() -> None:
    row = next(r for r in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if r.capability == "MSE")
    assert row.coverage is Gr10CoverageStatus.PARTIAL
    assert gr10_matrix_orchestration_status("MSE") is Gr10CoverageStatus.PARTIAL
