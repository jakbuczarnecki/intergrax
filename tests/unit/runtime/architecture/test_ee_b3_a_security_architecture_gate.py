# © Artur Czarnecki. All rights reserved.

"""EE-B3-A — security architecture documentation and module gate."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_THREAT_MODEL = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_SECURITY_BOUNDARY_AND_THREAT_MODEL.md"
)
_CERT = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B3_A_EXECUTION_SECURITY_BOUNDARY_THREAT_MODEL_CERTIFICATION.md"
)

_REQUIRED_SECTIONS = (
    "## 1. Protected assets",
    "## 2. Trust boundaries",
    "## 3. Principal model",
    "## 14. Security owner matrix",
    "## 15. Threat matrix (summary)",
    "## 19. Cross-session NPSC-5F",
)

_EE_B3_A_TESTS = (
    "tests/unit/runtime/architecture/test_ee_b3_a_identity_spoofing_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_cross_tenant_execution_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_authority_escalation_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_governance_bypass_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_child_authority_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_tool_authorization_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_retry_recovery_security_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_security_architecture_gate.py",
)


def test_ee_b3_a_threat_model_document_complete() -> None:
    assert _THREAT_MODEL.is_file()
    text = _THREAT_MODEL.read_text(encoding="utf-8")
    for heading in _REQUIRED_SECTIONS:
        assert heading in text, f"missing {heading!r}"
    assert "Identity ≠ Authority" in text or "Identity" in text


def test_ee_b3_a_qualification_document_and_npsc5f_handoff() -> None:
    assert _CERT.is_file()
    text = _CERT.read_text(encoding="utf-8")
    assert "NPSC-5F SECURITY FINDINGS HANDOFF" in text


def test_ee_b3_a_all_gate_modules_present() -> None:
    missing = [rel for rel in _EE_B3_A_TESTS if not (_REPO / rel).is_file()]
    assert missing == []
