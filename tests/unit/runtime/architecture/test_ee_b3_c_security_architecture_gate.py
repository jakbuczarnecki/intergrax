# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — adversarial certification documentation and static gates."""

from __future__ import annotations

import ast
import re
import pytest

from testing_support.security.abuse_case_fixtures import REPO_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ABUSE_MODEL = (
    REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_SECURITY_ADVERSARIAL_ABUSE_MODEL.md"
)
_CERT = (
    REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B3_C_SECURITY_ADVERSARIAL_ABUSE_CERTIFICATION.md"
)

_EE_B3_C_TESTS = (
    "tests/unit/runtime/architecture/test_ee_b3_c_identity_forgery_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_cross_tenant_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_child_authority_escalation_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_governance_spoofing_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_tool_side_effect_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_retry_recovery_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_checkpoint_tampering_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_confused_deputy_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_resource_exhaustion_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_compound_security_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_hitl_misuse_abuse.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_security_architecture_gate.py",
)

_REQUIRED_SECTIONS = (
    "## 1. Methodology",
    "## 2. Abuse-case matrix",
    "## 14. Cross-session exclusions",
    "## 13. Security bypass inventory",
)


def test_ee_b3_c_abuse_model_document_complete() -> None:
    assert _ABUSE_MODEL.is_file()
    text = _ABUSE_MODEL.read_text(encoding="utf-8")
    for heading in _REQUIRED_SECTIONS:
        assert heading in text, f"missing {heading!r}"


def test_ee_b3_c_qualification_document_present() -> None:
    assert _CERT.is_file()
    text = _CERT.read_text(encoding="utf-8")
    assert "START_HEAD" in text
    assert "NPSC-5F SECURITY HANDOFF" in text


def test_ee_b3_c_all_abuse_modules_present() -> None:
    missing = [rel for rel in _EE_B3_C_TESTS if not (REPO_ROOT / rel).is_file()]
    assert missing == []


def test_ee_b3_c_production_never_imports_testing_support_security() -> None:
    intergrax_root = REPO_ROOT / "intergrax"
    hits: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        if "testing_support.security" not in source:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            hits.append(
                f"{path.relative_to(REPO_ROOT)}:unparseable_with_testing_support_ref"
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("testing_support.security"):
                        hits.append(
                            f"{path.relative_to(REPO_ROOT)}:import {alias.name}"
                        )
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("testing_support.security"):
                    hits.append(f"{path.relative_to(REPO_ROOT)}:from {node.module}")
    assert hits == []


def test_ee_b3_c_p0_supported_production_bypass_zero() -> None:
    p0 = (
        REPO_ROOT
        / "docs"
        / "project"
        / "maintainers"
        / "qualification"
        / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
    )
    text = p0.read_text(encoding="utf-8")
    match = re.search(
        r"^\| Supported execution bypasses \(production\) \| (\d+) \|",
        text,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert int(match.group(1)) == 0
