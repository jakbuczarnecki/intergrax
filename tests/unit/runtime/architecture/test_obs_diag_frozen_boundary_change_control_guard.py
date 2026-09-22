# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-FREEZE-1 — frozen OBS/DIAG boundary change-control guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.obs_diag_freeze.authority_scan import (
    collect_authority_constructor_violations_in_source,
    collect_production_authority_constructor_violations,
)
from testing_support.obs_diag_freeze.guard import (
    missing_frozen_contract_modules,
    missing_reused_gate_modules,
    run_pytest_modules,
    stale_certification_record_referenced_in_freeze_support,
    verify_certification_provenance,
)
from testing_support.obs_diag_freeze.manifest import (
    CERTIFICATION_METADATA_RECONCILIATION_COMMIT,
    CERTIFICATION_RECORD_COMMIT,
    CERTIFIED_CODE_SHA,
    OBS_DIAG_A1_ARCHITECTURE_REGRESSION_TARGETS,
    OBS_DIAG_FROZEN_REUSED_GATE_MODULES,
    ObsDiagChangeClass,
    ObsDiagFrozenInvariantId,
    ObsDiagRequalificationSignal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_freeze]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_obs_diag_freeze_manifest_certification_shas() -> None:
    assert CERTIFIED_CODE_SHA == "c5acfc17ecea4d0e4b03405eabf735799c020ff8"
    assert CERTIFICATION_RECORD_COMMIT == "150cf9a51833a1cf2ca89ded7e9447f3f9e2eea7"
    assert CERTIFICATION_METADATA_RECONCILIATION_COMMIT == (
        "f34c55accf23fa46633ded1803150de94f37770a"
    )


def test_obs_diag_freeze_invariants_and_change_classes_defined() -> None:
    assert len(ObsDiagFrozenInvariantId) >= 10
    assert ObsDiagChangeClass.SAFE_EXTENSION.value == "safe_extension"
    assert ObsDiagRequalificationSignal.NEW_AUTHORITY.value == "new_authority"


def test_obs_diag_freeze_certification_provenance_is_ancestor_of_head() -> None:
    result = verify_certification_provenance(_REPO_ROOT)
    assert result.certified_code_is_ancestor
    assert result.certification_record_is_ancestor


def test_obs_diag_freeze_no_stale_certification_record_in_support_tree() -> None:
    assert stale_certification_record_referenced_in_freeze_support(_REPO_ROOT) == ()


def test_obs_diag_freeze_frozen_contract_modules_present() -> None:
    assert missing_frozen_contract_modules(_REPO_ROOT) == ()


def test_obs_diag_freeze_reused_gate_modules_exist() -> None:
    assert missing_reused_gate_modules(_REPO_ROOT) == ()


def test_obs_diag_freeze_production_authority_constructor_sites() -> None:
    violations = collect_production_authority_constructor_violations(_REPO_ROOT)
    assert violations == ()


def test_obs_diag_freeze_negative_proof_second_diagnostic_orchestrator() -> None:
    source = """
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator

def illegal_second_authority() -> DiagnosticOrchestrator:
    return DiagnosticOrchestrator()
"""
    rel = "applications/example/illegal_second_orchestrator_authority.py"
    violations = collect_authority_constructor_violations_in_source(source, rel_path=rel)
    assert len(violations) == 1
    assert violations[0].symbol == "DiagnosticOrchestrator"
    assert violations[0].change_class == ObsDiagChangeClass.REQUALIFICATION_REQUIRED.value


def test_obs_diag_freeze_extension_proof_legal_plugin_not_blocked() -> None:
    fixture = (
        _REPO_ROOT
        / "testing_support/obs_diag_freeze/fixtures/legal_plugin_grouping_strategy.py"
    )
    source = fixture.read_text(encoding="utf-8")
    rel = fixture.relative_to(_REPO_ROOT).as_posix()
    violations = collect_authority_constructor_violations_in_source(source, rel_path=rel)
    assert violations == ()


@pytest.mark.obs_diag_freeze_bundle
def test_obs_diag_freeze_reused_architecture_gates_pass() -> None:
    proc = run_pytest_modules(_REPO_ROOT, OBS_DIAG_FROZEN_REUSED_GATE_MODULES)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.obs_diag_freeze_bundle
def test_obs_diag_freeze_a1_architecture_regression_subset_pass() -> None:
    proc = run_pytest_modules(_REPO_ROOT, OBS_DIAG_A1_ARCHITECTURE_REGRESSION_TARGETS)
    assert proc.returncode == 0, proc.stdout + proc.stderr
