# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-CONFORMANCE — qualification manifest and proof module registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class _ConformanceProof:
    scenario: str
    modules: tuple[str, ...]


_P1_PROOFS: tuple[_ConformanceProof, ...] = (
    _ConformanceProof(
        "successful_execution_no_problem",
        ("tests/integration/runtime/test_obs_diag_conformance_e2e.py",),
    ),
    _ConformanceProof(
        "terminal_failure_problem",
        ("tests/integration/runtime/test_obs_diag_conformance_e2e.py",),
    ),
    _ConformanceProof(
        "retry_a1_a2_reconstruction",
        ("tests/integration/runtime/test_obs_diag_conformance_e2e.py",),
    ),
    _ConformanceProof(
        "child_lineage_via_shared_reconstruction",
        ("tests/integration/runtime/test_obs_diag_conformance_e2e.py",),
    ),
    _ConformanceProof(
        "multi_execution_grouping",
        ("tests/integration/runtime/test_obs_diag_conformance_e2e.py",),
    ),
    _ConformanceProof(
        "tenant_isolation",
        (
            "tests/integration/runtime/test_obs_diag_conformance_e2e.py",
            "tests/unit/runtime/diagnostics/test_diagnostic_orchestrator.py",
        ),
    ),
    _ConformanceProof(
        "signal_subject_separation",
        (
            "tests/integration/runtime/test_obs_diag_conformance_e2e.py",
            "tests/unit/runtime/diagnostics/test_application_diagnostic_subjects.py",
        ),
    ),
    _ConformanceProof(
        "architecture_gates",
        ("tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py",),
    ),
    _ConformanceProof(
        "reconstruction_ownership",
        ("tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py",),
    ),
    _ConformanceProof(
        "terminal_ordering_port",
        ("tests/unit/runtime/architecture/test_obs_diag_port_1_gates.py",),
    ),
    _ConformanceProof(
        "grouping_pluginability",
        ("tests/unit/runtime/diagnostics/test_problem_grouping.py",),
    ),
    _ConformanceProof(
        "diagnostic_read_persisted_truth",
        ("tests/unit/runtime/diagnostics/test_diagnostic_read_service.py",),
    ),
)


def test_p1_proof_modules_exist_on_disk() -> None:
    missing: list[str] = []
    for proof in _P1_PROOFS:
        for module in proof.modules:
            path = _REPO_ROOT / module
            if not path.is_file():
                missing.append(module)
    assert missing == []


def test_obs_diag_conformance_pytest_mark_registered() -> None:
    ini = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "obs_diag_conformance" in ini
