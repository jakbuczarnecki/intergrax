# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — architecture gates (single owner, forbidden worker planes)."""

from __future__ import annotations

import re
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
    / "EXECUTION_ENGINE_WORKER_ISOLATION_AND_FAILURE_CONTAINMENT_MODEL.md"
)
_CERT = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B1_3_WORKER_ISOLATION_FAILURE_CONTAINMENT_ENTERPRISE_CERTIFICATION.md"
)
_CONCURRENT = (
    _REPO / "intergrax" / "runtime" / "execution" / "concurrent_execution_work.py"
)

_FORBIDDEN = (
    "WorkerRuntime",
    "WorkerSupervisorRuntime",
    "WorkerScheduler",
    "WorkerRetryLoop",
    "WorkerRecoveryEngine",
    "ExecutionWorkerQueue",
    "AlternativeExecutionPool",
    "WorkerManager",
    "WorkerRegistry",
    "WorkerLifecycleEngine",
)

_B1_3_TESTS = (
    "tests/unit/runtime/architecture/test_ee_b1_3_worker_isolation_contract.py",
    "tests/unit/runtime/architecture/test_ee_b1_3_worker_exception_containment.py",
    "tests/unit/runtime/architecture/test_ee_b1_3_worker_timeout_containment.py",
    "tests/unit/runtime/architecture/test_ee_b1_3_worker_cancellation_containment.py",
    "tests/unit/runtime/architecture/test_ee_b1_3_worker_capacity_release.py",
    "tests/unit/runtime/architecture/test_ee_b1_3_worker_failure_storm.py",
    "tests/unit/runtime/architecture/test_ee_b1_3_child_execution_failure_containment.py",
)


def test_ee_b1_3_documents_present() -> None:
    assert _MODEL.is_file()
    assert _CERT.is_file()


def test_ee_b1_3_gate_modules_exist() -> None:
    for rel in _B1_3_TESTS:
        assert (_REPO / rel).is_file(), rel


def test_ee_b1_3_no_forbidden_worker_plane_symbols_in_concurrent_work() -> None:
    text = _CONCURRENT.read_text(encoding="utf-8")
    hits = [sym for sym in _FORBIDDEN if sym in text]
    assert hits == []


def test_ee_b1_3_concurrent_work_no_governance_imports() -> None:
    text = _CONCURRENT.read_text(encoding="utf-8")
    assert "PolicyEngine" not in text
    assert "GovernanceEngine" not in text


def test_ee_b1_3_concurrent_work_no_identity_mint() -> None:
    mint_pattern = re.compile(
        r"\b(mint_run_id|mint_attempt_id|mint_execution_id|mint_task_id)\s*\(",
    )
    text = _CONCURRENT.read_text(encoding="utf-8")
    assert mint_pattern.search(text) is None


def test_ee_b1_3_canonical_owner_named_in_model() -> None:
    text = _MODEL.read_text(encoding="utf-8")
    assert "execute_concurrent_execution_work_resilient" in text
    assert (
        "Who owns worker failure containment" in text
        or "who owns worker failure" in text.lower()
    )
