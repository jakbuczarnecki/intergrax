# © Artur Czarnecki. All rights reserved.

"""DS-MIG-03 — production runtime/critic must not retain L2 human verification."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CRITIC_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "critic"

_FORBIDDEN_FRAGMENTS = (
    "L2Gateway",
    "L2_HUMAN",
    "CriticAction.ESCALATE_HITL",
    "l2_human_required",
    "l2_borderline_margin",
    "l2_gateway",
)


def test_l2_gateway_file_removed() -> None:
    assert not (_CRITIC_ROOT / "l2_gateway.py").is_file()


def test_runtime_critic_has_no_l2_human_symbols() -> None:
    violations: list[str] = []
    for path in _CRITIC_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for fragment in _FORBIDDEN_FRAGMENTS:
            if fragment in source:
                violations.append(f"{rel}: {fragment}")
    assert violations == [], (
        "runtime/critic must not contain L2/HITL verification symbols: "
        + ", ".join(violations)
    )
