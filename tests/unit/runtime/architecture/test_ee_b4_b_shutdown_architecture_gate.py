# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — graceful shutdown architecture and import gates."""

from __future__ import annotations

import ast
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
    / "EXECUTION_ENGINE_GRACEFUL_SHUTDOWN_DRAIN_TERMINATION_MODEL.md"
)
_CERT = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B4_B_GRACEFUL_SHUTDOWN_DRAIN_TERMINATION_CERTIFICATION.md"
)
_SUPPORT = _REPO / "testing_support" / "shutdown"
_INTERGRAX = _REPO / "intergrax"

_EE_B4_B_TESTS = (
    "tests/unit/runtime/architecture/test_ee_b4_b_stop_accepting_new_work.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_active_execution_drain.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_shutdown_admission_race.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_worker_failure_during_drain.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_cancellation_during_drain.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_mandatory_evidence_flush.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_final_state_persistence.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_capacity_release.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_worker_task_leak.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_shutdown_idempotency.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_concurrent_shutdown.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_compound_shutdown_failure.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_root_vs_child_drain.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_shutdown_health_semantics.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_shutdown_architecture_gate.py",
)

_FORBIDDEN_PRODUCTION = (
    "ShutdownRuntime",
    "DrainRuntime",
    "WorkerShutdownManager",
    "ExecutionTerminator",
    "LifecycleSupervisorV2",
)

_NPSC5F_PROTECTED = (
    "runtime/observability/causal_evidence.py",
    "runtime/observability/causal_evidence_export.py",
    "runtime/observability/export_boundary.py",
)


def test_ee_b4_b_documents_present() -> None:
    assert _MODEL.is_file()
    assert _CERT.is_file()


def test_ee_b4_b_test_modules_present() -> None:
    for rel in _EE_B4_B_TESTS:
        assert (_REPO / rel).is_file(), rel


def test_ee_b4_b_support_package_present() -> None:
    assert (_SUPPORT / "reference_lifecycle.py").is_file()


def test_ee_b4_b_no_forbidden_shutdown_runtime_in_production() -> None:
    hits: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_PRODUCTION:
            if symbol in text:
                hits.append(f"{path.relative_to(_INTERGRAX)}:{symbol}")
    assert hits == []


def test_ee_b4_b_production_does_not_import_shutdown_testing_support() -> None:
    offenders: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("testing_support.shutdown"):
                    offenders.append(path.relative_to(_INTERGRAX).as_posix())
    assert offenders == []


def test_ee_b4_b_npsc5f_protected_surfaces_unmodified_in_task_scope() -> None:
    for rel in _NPSC5F_PROTECTED:
        assert (_INTERGRAX / rel).is_file()
