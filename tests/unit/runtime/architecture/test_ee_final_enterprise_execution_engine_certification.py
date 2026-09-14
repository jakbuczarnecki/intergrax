# © Artur Czarnecki. All rights reserved.

"""EE-FINAL — enterprise Execution Engine cross-session certification architecture gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.unit.runtime.architecture._ee_final_enterprise_facts import (
    EE_B2_FINAL_QUALIFICATION,
    EE_FINAL_ARCH_QUALIFICATION,
    FINAL_ANCHOR_COMMITS,
    FINAL_ARCHITECTURE_DOC,
    FINAL_GATE_MODULES,
    FINAL_QUALIFICATION_DOC,
    P0_INVENTORY,
    PLATFORM_REVALIDATION_QUALIFICATION,
)
from tests.unit.runtime.architecture._ee_final_arch_facts import p0_bypass_count

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]


def _is_ancestor(ancestor: str, descendant: str = "HEAD") -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=_REPO,
        check=False,
    )
    return proc.returncode == 0


def test_ee_final_enterprise_documents_present() -> None:
    assert FINAL_ARCHITECTURE_DOC.is_file()
    assert FINAL_QUALIFICATION_DOC.is_file()
    assert EE_FINAL_ARCH_QUALIFICATION.is_file()
    assert EE_B2_FINAL_QUALIFICATION.is_file()
    assert PLATFORM_REVALIDATION_QUALIFICATION.is_file()
    assert P0_INVENTORY.is_file()


def test_ee_final_enterprise_p0_zero_bypass_ssot() -> None:
    assert p0_bypass_count() == 0


def test_ee_final_enterprise_qualification_lists_all_anchor_commits() -> None:
    text = FINAL_QUALIFICATION_DOC.read_text(encoding="utf-8")
    for label, sha in FINAL_ANCHOR_COMMITS:
        assert sha in text, f"missing anchor SHA for {label}"


def test_ee_final_enterprise_all_anchors_are_ancestors_of_head() -> None:
    missing = [label for label, sha in FINAL_ANCHOR_COMMITS if not _is_ancestor(sha)]
    assert missing == []


def test_ee_final_enterprise_representative_gate_modules_exist() -> None:
    for rel in FINAL_GATE_MODULES:
        assert (_REPO / rel).is_file(), rel
    arch_gates = sorted(
        _REPO.glob("tests/unit/runtime/architecture/test_ee_final_arch_*.py")
    )
    assert len(arch_gates) >= 10


def test_ee_final_enterprise_architecture_doc_has_required_sections() -> None:
    text = FINAL_ARCHITECTURE_DOC.read_text(encoding="utf-8")
    required = (
        "Canonical execution path",
        "Ownership matrix",
        "Identity model",
        "Authority",
        "Governance",
        "Nexus",
        "Child execution",
        "RuntimeToolInvoker",
        "Retry",
        "Recovery",
        "Evidence",
        "Reconstruction",
        "Observability",
        "Diagnostics",
        "Security",
        "Shutdown",
        "Operational readiness",
        "Runbooks",
        "Pluginability",
        "Persistence abstraction",
        "Scale model",
        "Explicit non-goals",
        "zero-bypass",
    )
    for heading in required:
        assert heading.lower() in text.lower(), heading


def test_ee_final_enterprise_freeze_statement_present() -> None:
    text = FINAL_QUALIFICATION_DOC.read_text(encoding="utf-8")
    assert "EXECUTION ENGINE STAGE" in text
    assert "single authoritative execution subsystem" in text.lower()


@pytest.mark.gate
def test_ee_final_enterprise_no_execution_core_drift_since_revalidation() -> None:
    proc = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "953a38a1c6f52ca4dec2c55b18942ba75c97854b..HEAD",
            "--",
            "intergrax/runtime/execution",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
