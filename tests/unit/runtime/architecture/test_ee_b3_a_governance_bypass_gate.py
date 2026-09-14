# © Artur Czarnecki. All rights reserved.

"""EE-B3-A — governance bypass and second security engine gate."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_P0 = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
)
_INTERGRAX = _REPO / "intergrax"

_FORBIDDEN_ENGINES = (
    "SecurityRuntime",
    "SecurityPolicyEngine",
    "SecurityScheduler",
    "AuthorizationRuntime",
)


def test_ee_b3_a_p0_supported_production_bypass_zero() -> None:
    text = _P0.read_text(encoding="utf-8")
    match = re.search(
        r"^\| Supported execution bypasses \(production\) \| (\d+) \|",
        text,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert int(match.group(1)) == 0


def test_ee_b3_a_no_second_security_engine_symbols() -> None:
    hits: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        body = path.read_text(encoding="utf-8")
        for sym in _FORBIDDEN_ENGINES:
            if sym in body:
                hits.append(f"{path.relative_to(_REPO)}:{sym}")
    assert hits == []


def test_ee_b3_a_policy_admission_evaluator_uses_runtime_policy_engine() -> None:
    source = (
        _REPO
        / "intergrax"
        / "runtime"
        / "governance"
        / "runtime_execution_policy_admission.py"
    )
    text = source.read_text(encoding="utf-8")
    assert "RuntimePolicyEngine" in text
    assert "evaluate_root_execution_admission" in text
