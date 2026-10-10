# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1 ambiguous pin outcome architecture gates (docs-only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_START_HEAD = "2e337c4b04c112d118aa0a4817be2ea13c08ef39"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_R1_AMBIGUOUS_PIN_OUTCOME_STAGING_TIMESTAMP_SEMANTICS.md"
)
_PRIOR_LOCK = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_P2_PIN_RECOVERY_STAGING_CONTRACT_LOCK.md"
)
_ROADMAP = _REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
_PINNING_PROTOCOL = (
    _REPO_ROOT / "intergrax/integrations/contracts/execution_integration_configuration_pinning.py"
)
_CONTRACT_TESTS = (
    _REPO_ROOT
    / "tests/unit/applications/integrations/"
    "test_trace_x_p5_r2_p4_r1_r1_r1_r1_r1_r1_pin_ambiguous_outcome_contract.py"
)


def test_txp5r2p4r1r1r1r1r1r1r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p4r1r1r1r1r1r1r1_q02_ambiguous_outcome_lock_ready_for_audit() -> None:
    assert _LOCK_DOC.is_file()
    text = _LOCK_DOC.read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in text
    assert "R2-P4-P2-PIN-AMBIGUOUS-COMMIT-OUTCOME-31" in text
    assert "requirement_boundary_prepared_at" in text
    assert "read_pin_records" in text
    assert "find_pin_record_for_subject" in text
    assert "recovery reconciliation" in text.lower() or "Recovery reconciliation" in text
    assert "FRZ-TRC-11" in text and "OPEN" in text
    assert "Production delta" in text and "**0**" in text


def test_txp5r2p4r1r1r1r1r1r1r1_q03_prior_p2_lock_superseded_for_ambiguous_outcome() -> None:
    prior = _PRIOR_LOCK.read_text(encoding="utf-8")
    assert "BLOCKED / SUPERSEDED BY CHILD" in prior
    assert "AMBIGUOUS_PIN_OUTCOME_STAGING_TIMESTAMP_SEMANTICS" in prior
    assert "R2-P4-P2-PIN-AMBIGUOUS-COMMIT-OUTCOME-31" in prior


def test_txp5r2p4r1r1r1r1r1r1r1_q04_roadmap_registers_child_trace_id() -> None:
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    assert "TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1" in roadmap
    assert "AMBIGUOUS_PIN_OUTCOME_STAGING_TIMESTAMP_SEMANTICS" in roadmap


def test_txp5r2p4r1r1r1r1r1r1r1_q05_no_second_staging_store_in_pinning_protocol() -> None:
    source = _PINNING_PROTOCOL.read_text(encoding="utf-8")
    assert "ExecutionIntegrationConfigurationPinningStore" in source
    lowered = source.lower()
    assert "recovery_staging_store" not in lowered
    assert "shadow" not in lowered or "shadow index" not in lowered


def test_txp5r2p4r1r1r1r1r1r1r1_q06_contract_tests_module_present() -> None:
    assert _CONTRACT_TESTS.is_file()
    body = _CONTRACT_TESTS.read_text(encoding="utf-8")
    assert "lost_acknowledgement" in body or "lost_ack" in body
    assert "concurrent" in body.lower()
