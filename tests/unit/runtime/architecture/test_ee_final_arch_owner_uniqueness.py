# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — single authoritative owner per execution concern."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import (
    ARCH_MODEL,
    OWNERSHIP_MODEL,
    _REPO_ROOT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_OWNER_SYMBOLS = (
    "ExecutionRuntime",
    "ExecutionIdentityAuthority",
    "ChildExecutionRunner",
    "RuntimeToolInvoker",
    "GraphExecutor",
)


def test_ee_final_arch_owner_matrix_documented() -> None:
    text = ARCH_MODEL.read_text(encoding="utf-8")
    assert "## 2. Final owner matrix" in text
    for symbol in _OWNER_SYMBOLS:
        assert symbol in text


def test_ee_final_arch_ee_a1_ownership_model_still_certified() -> None:
    assert OWNERSHIP_MODEL.is_file()
    ownership = OWNERSHIP_MODEL.read_text(encoding="utf-8")
    assert "ExecutionRuntime" in ownership
    gate = (
        _REPO_ROOT
        / "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py"
    )
    assert gate.is_file()
