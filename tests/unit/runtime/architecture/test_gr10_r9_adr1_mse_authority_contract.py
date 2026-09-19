# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-ADR1 — canonical MSE authority contract ADR gates (post R9-R1 baseline)."""

from __future__ import annotations

import inspect
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
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ADR_PATH = (
    REPO_ROOT
    / "docs/project/technical/adr/entries/2026-09-19/ADR-GR-10-002.md"
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
    assert "`object`" in adr
    assert "no synthetic membership" in adr.lower() or "Synthetic `WorkspaceMembership`" in adr
    assert "no default ALLOW" in adr.lower() or "Default `PolicyAction.ALLOW`" in adr
    assert "InMemory" in adr


def test_gr10_r9_adr1_canonical_port_typed_return() -> None:
    sig = inspect.signature(MeaningfulSideEffectAuthorizationPort.authorize)
    assert "MeaningfulSideEffectAuthorizationResult" in str(sig.return_annotation)
    assert "object" not in str(sig.return_annotation)


def test_gr10_r9_adr1_boundary_authorize_is_typed_at_runtime_impl() -> None:
    sig = inspect.signature(MeaningfulSideEffectAuthorizationBoundary.authorize)
    assert sig.return_annotation is not object
    assert "MeaningfulSideEffectAuthorizationResult" in str(sig.return_annotation)


def test_gr10_r9_adr1_matrix_honesty_orchestration_mse_qualified() -> None:
    row = next(r for r in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if r.capability == "MSE")
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("MSE") is Gr10CoverageStatus.QUALIFIED
