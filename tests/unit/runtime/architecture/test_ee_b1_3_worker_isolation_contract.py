# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — concurrent work contract and architecture doc presence."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    ConcurrentExecutionWorkOutcome,
)

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

_REQUIRED_SECTIONS = (
    "## 3. Canonical worker failure containment owner",
    "## 4. Strict vs resilient semantics",
    "## 6. Timeout semantics",
    "## 7. Cancellation semantics",
    "## 8. Capacity interaction (EE-B1.2)",
    "## 9. Retry & recovery separation",
    "## 10. Child execution behavior",
    "## 11. Fan-out & GraphExecutor",
    "## 17. Shutdown interaction",
    "## 18. Deadlock analysis",
)


def test_ee_b1_3_worker_isolation_model_document_present() -> None:
    assert _MODEL.is_file()
    text = _MODEL.read_text(encoding="utf-8")
    for heading in _REQUIRED_SECTIONS:
        assert heading in text, f"missing {heading}"
    assert "concurrent_execution_work.py" in text


def test_ee_b1_3_concurrent_outcome_typed_invariants() -> None:
    ok = ConcurrentExecutionWorkOutcome(
        disposition=ConcurrentExecutionWorkDisposition.SUCCEEDED,
        result=42,
        error=None,
    )
    assert ok.result == 42
    with pytest.raises(ValueError, match="failed outcome must carry error"):
        ConcurrentExecutionWorkOutcome(
            disposition=ConcurrentExecutionWorkDisposition.FAILED,
            result=None,
            error=None,
        )
