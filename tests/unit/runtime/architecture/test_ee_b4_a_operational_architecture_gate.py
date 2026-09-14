# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — operational readiness architecture gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_MODEL = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_OPERATIONAL_READINESS_SLO_HEALTH_MODEL.md"
)
_CERT = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B4_A_OPERATIONAL_READINESS_SLO_HEALTH_CERTIFICATION.md"
)
_SUPPORT = _REPO / "testing_support" / "execution_operational_readiness"
_INTERGRAX = _REPO / "intergrax"

_EE_B4_A_TESTS = (
    "tests/unit/runtime/architecture/test_ee_b4_a_health_state_model.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_readiness_semantics.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_liveness_semantics.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_capacity_saturation_readiness.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_mandatory_evidence_readiness.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_observability_degradation.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_shutdown_readiness.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_policy_denial_health.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_sli_slo_contract.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_operational_architecture_gate.py",
)

_FORBIDDEN_PRODUCTION = (
    "HealthRuntime",
    "MetricsRuntime",
    "SloEngine",
    "OperationalEventBus",
    "ReadinessScheduler",
    "HealthStateMachineRuntime",
)

_NPSC5F_PROTECTED = (
    "runtime/observability/causal_evidence.py",
    "runtime/observability/causal_evidence_export.py",
    "runtime/observability/export_boundary.py",
)


def test_ee_b4_a_documents_present() -> None:
    assert _MODEL.is_file()
    assert _CERT.is_file()


def test_ee_b4_a_test_modules_present() -> None:
    for rel in _EE_B4_A_TESTS:
        assert (_REPO / rel).is_file(), rel


def test_ee_b4_a_support_package_present() -> None:
    assert (_SUPPORT / "assessment.py").is_file()
    assert (_SUPPORT / "sli_catalog.py").is_file()


def test_ee_b4_a_no_forbidden_operational_runtime_in_production() -> None:
    hits: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        rel = path.relative_to(_INTERGRAX).as_posix()
        if any(p in rel for p in _NPSC5F_PROTECTED):
            continue
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_PRODUCTION:
            if token in text:
                hits.append(f"{rel}: {token}")
    assert hits == []


def test_ee_b4_a_model_documents_cross_session_exclusions() -> None:
    text = _MODEL.read_text(encoding="utf-8")
    assert "causal_evidence" in text
    assert "background_execution" in text
