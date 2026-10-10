# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P6 closed-world mechanical gates (FRZ-TRC-09 / FRZ-TRC-10)."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p6_adversarial_matrix import (
    P6_SEMANTIC_OWNER_MATRIX,
    TRACE_X_P6_START_HEAD,
)
from tests.qualification.trace_x._trace_x_p6_discovery import (
    discover_restart_resume_path_keys,
    discover_terminal_producer_keys,
    grep_production_pattern,
)
from tests.qualification.trace_x._trace_x_p6_restart_resume_classification import (
    RESTART_RESUME_REGISTRY,
    classify_restart_resume_path,
    compare_restart_resume_registry,
)
from tests.qualification.trace_x._trace_x_p6_semantic_owner_discovery import (
    compare_semantic_owner_gate,
    discover_owner_candidates,
)
from tests.qualification.trace_x._trace_x_p6_terminal_producer_classification import (
    TERMINAL_PRODUCER_REGISTRY,
    classify_terminal_producer,
    compare_terminal_producer_registry,
)
from tests.qualification.trace_x._trace_x_p6_types import (
    RestartResumePathClass,
    TerminalProducerRole,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TERMINAL_SERVICE = (
    _REPO_ROOT / "intergrax/runtime/execution/execution_terminal/service.py"
)
_DIAGNOSTICS = _REPO_ROOT / "intergrax/runtime/diagnostics"


def test_txp6_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P6_START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp6_q02_restart_resume_closed_world_parity() -> None:
    discovered = discover_restart_resume_path_keys()
    result = compare_restart_resume_registry(discovered, RESTART_RESUME_REGISTRY)
    assert not result.unknown, f"unclassified discovery: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan registry: {sorted(result.orphan)}"
    assert not result.production_bypass, f"bypass: {sorted(result.production_bypass)}"
    assert not result.unclassified, f"G-class: {sorted(result.unclassified)}"
    assert result.ok
    assert len(discovered) >= 1


def test_txp6_q03_terminal_producer_closed_world_parity() -> None:
    discovered = discover_terminal_producer_keys()
    result = compare_terminal_producer_registry(discovered, TERMINAL_PRODUCER_REGISTRY)
    assert not result.unknown, f"unclassified discovery: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan registry: {sorted(result.orphan)}"
    assert not result.forbidden_bypass, f"bypass: {sorted(result.forbidden_bypass)}"
    assert not result.unclassified, f"unclear: {sorted(result.unclassified)}"
    assert result.ok


def test_txp6_q04_exactly_one_terminal_truth_owner() -> None:
    truth = [
        row.path
        for row in TERMINAL_PRODUCER_REGISTRY
        if row.role is TerminalProducerRole.CANONICAL_TERMINAL_TRUTH
    ]
    assert truth == ["intergrax/runtime/execution/execution_terminal/service.py"]


def test_txp6_q05_diagnostics_terminal_authority_zero() -> None:
    forbidden = (
        "commit_terminal_outcome",
        "ExecutionTerminalService(",
        "record_cancellation(",
    )
    for path in _DIAGNOSTICS.rglob("*.py"):
        if "/tests/" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: diagnostics must not mint terminal truth ({token})"


def test_txp6_q06_diagnostics_resume_authority_zero() -> None:
    hits = grep_production_pattern(r"diagnostics.*resume|resume.*diagnostic_orchestrator")
    diag_hits = [h for h in hits if h.startswith("intergrax/runtime/diagnostics/")]
    assert not diag_hits


def test_txp6_q07_single_execution_terminal_service_class() -> None:
    tree = ast.parse(_TERMINAL_SERVICE.read_text(encoding="utf-8"))
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExecutionTerminalService"
    ]
    assert len(classes) == 1


def test_txp6_q08_no_unclear_restart_resume_paths() -> None:
    unclear = [
        row.path
        for row in RESTART_RESUME_REGISTRY
        if row.classification is RestartResumePathClass.G_UNCLEAR
    ]
    assert unclear == []


def test_txp6_q09_no_production_bypass_restart_resume() -> None:
    bypass = [
        row.path
        for row in RESTART_RESUME_REGISTRY
        if row.classification is RestartResumePathClass.F_PRODUCTION_BYPASS
    ]
    assert bypass == []


def test_txp6_q10_semantic_owner_discovery_exact_parity() -> None:
    for concern, expected_owners in P6_SEMANTIC_OWNER_MATRIX:
        result = compare_semantic_owner_gate(concern)
        assert result.ok, (
            concern,
            result.expected_owner_set,
            result.discovered_owner_set,
        )
        assert discover_owner_candidates(concern) == expected_owners
        assert all(owner.strip() for owner in result.discovered_owner_set)


def test_txp6_q15_negative_sensitivity_terminal_truth_duplicate_owner() -> None:
    concern = "terminal state truth owner"
    baseline = compare_semantic_owner_gate(concern)
    assert baseline.ok
    synthetic = frozenset({"intergrax/runtime/synthetic_p6_duplicate_terminal_truth_owner.py"})
    failed = compare_semantic_owner_gate(concern, extra_discovered=synthetic)
    assert synthetic.issubset(failed.discovered_owner_set)
    assert not failed.ok


def test_txp6_q16_negative_sensitivity_resume_coordination_duplicate_owner() -> None:
    concern = "resume coordination owner"
    baseline = compare_semantic_owner_gate(concern)
    assert baseline.ok
    synthetic = frozenset({"intergrax/runtime/synthetic_p6_duplicate_resume_coordinator.py"})
    failed = compare_semantic_owner_gate(concern, extra_discovered=synthetic)
    assert not failed.ok


def test_txp6_q17_negative_sensitivity_retry_orchestration_duplicate_owner() -> None:
    concern = "retry orchestration owner"
    baseline = compare_semantic_owner_gate(concern)
    assert baseline.ok
    synthetic = frozenset({"intergrax/runtime/synthetic_p6_duplicate_retry_orchestrator.py"})
    failed = compare_semantic_owner_gate(concern, extra_discovered=synthetic)
    assert not failed.ok


def test_txp6_q11_negative_sensitivity_unregistered_restart_path() -> None:
    baseline = compare_restart_resume_registry(
        discover_restart_resume_path_keys(),
        RESTART_RESUME_REGISTRY,
    )
    assert baseline.ok
    synthetic = ("intergrax/runtime/example_p6_restart_surface.py", "module")
    failed = compare_restart_resume_registry(
        discover_restart_resume_path_keys() | {synthetic},
        RESTART_RESUME_REGISTRY,
    )
    assert synthetic in failed.unknown
    assert not failed.ok


def test_txp6_q12_restart_classifier_fail_closed_without_explicit_rule() -> None:
    synthetic_path = "intergrax/runtime/synthetic_p6_restart_classifier_probe.py"
    assert classify_restart_resume_path(synthetic_path) is RestartResumePathClass.G_UNCLEAR


def test_txp6_q13_terminal_classifier_fail_closed_without_explicit_rule() -> None:
    synthetic_path = "intergrax/runtime/synthetic_p6_terminal_classifier_probe.py"
    assert classify_terminal_producer(synthetic_path) is TerminalProducerRole.UNCLEAR


def test_txp6_q14_negative_sensitivity_unregistered_terminal_surface() -> None:
    baseline = compare_terminal_producer_registry(
        discover_terminal_producer_keys(),
        TERMINAL_PRODUCER_REGISTRY,
    )
    assert baseline.ok
    synthetic = ("intergrax/runtime/example_p6_terminal_surface.py", "module")
    failed = compare_terminal_producer_registry(
        discover_terminal_producer_keys() | {synthetic},
        TERMINAL_PRODUCER_REGISTRY,
    )
    assert synthetic in failed.unknown
    assert not failed.ok
