# © Artur Czarnecki. All rights reserved.

"""EE-B1.1 — Execution Engine reliability foundation & failure semantics certification."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RELIABILITY_MODEL = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_RELIABILITY_MODEL.md"
)
_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"
_AGENTS = _REPO_ROOT / "agents"
_INTEGRATIONS = _REPO_ROOT / "intergrax" / "integrations"
_APPLICATIONS = _REPO_ROOT / "applications"

_REQUIRED_SECTIONS = (
    "## 1. Purpose",
    "## 2. ETAP 0 — Reliability inventory (audit baseline)",
    "## 3. Reliability ownership",
    "## 4. Failure classification contract",
    "## 5. Persistence failure semantics",
    "## 6. Worker failure semantics",
    "## 7. Graceful shutdown contract",
)

_FORBIDDEN_SECOND_OWNER_NAMES = (
    "SecondExecutionRuntime",
    "SecondScheduler",
    "SecondRetryEngine",
    "SecondCheckpointManager",
)


def _py_files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


def test_ee_b1_1_reliability_model_document_present_and_structured() -> None:
    assert _RELIABILITY_MODEL.is_file()
    text = _RELIABILITY_MODEL.read_text(encoding="utf-8")
    for heading in _REQUIRED_SECTIONS:
        assert heading in text, f"missing section {heading!r}"
    assert "| execution lifecycle | `ExecutionRuntime` |" in text
    assert "ExecutionFailureSemanticCategory" in text or "failure_classification_contract" in text


def test_ee_b1_1_failure_classification_contract_importable() -> None:
    from intergrax.contracts.execution_reliability import (
        ExecutionFailureSemanticCategory,
    )
    from intergrax.runtime.execution.reliability import default_execution_failure_classifier

    classifier = default_execution_failure_classifier()
    decision = classifier.classify(
        __import__(
            "intergrax.contracts.execution_reliability",
            fromlist=["ExecutionFailureContext"],
        ).ExecutionFailureContext(timeout=True, reason="deadline"),
    )
    assert decision.category is ExecutionFailureSemanticCategory.TRANSIENT
    assert decision.retry_projection is not None


def test_ee_b1_1_no_second_execution_engine_owner_symbols() -> None:
    roots = (_EXECUTION_ROOT, _AGENTS, _INTEGRATIONS, _APPLICATIONS)
    hits: list[str] = []
    for root in roots:
        for path in _py_files_under(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            for symbol in _FORBIDDEN_SECOND_OWNER_NAMES:
                if symbol in text:
                    hits.append(f"{rel}: {symbol}")
    assert hits == [], f"forbidden second-owner symbols: {hits}"


def test_ee_b1_1_single_execution_runtime_class_in_canonical_module() -> None:
    runtime_py = _EXECUTION_ROOT / "runtime.py"
    text = runtime_py.read_text(encoding="utf-8")
    matches = re.findall(r"^class ExecutionRuntime\b", text, flags=re.MULTILINE)
    assert len(matches) == 1


def test_ee_b1_1_b1_gate_modules_exist() -> None:
    gates = (
        "tests/unit/runtime/architecture/test_ee_b1_1_worker_failure_isolation.py",
        "tests/unit/runtime/architecture/test_ee_b1_1_shutdown_contract.py",
        "tests/unit/runtime/architecture/test_ee_b1_1_persistence_failure_contract.py",
    )
    missing = [g for g in gates if not (_REPO_ROOT / g).is_file()]
    assert missing == []
