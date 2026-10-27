# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "ci"))

from run_regression_gate_ci import fast_steps, smoke_steps, split_command_chain  # noqa: E402


def test_split_command_chain_splits_on_and() -> None:
    segments = split_command_chain(["uv", "run", "&&", "uv", "lock"])
    assert segments == [["uv", "run"], ["uv", "lock"]]


def test_smoke_steps_without_sync_has_five_steps() -> None:
    steps = smoke_steps(include_uv_sync=False)
    assert len(steps) == 5
    names = [step.name for step in steps]
    assert names[-2] == "Tier boundary: agents reflection (smoke)"
    assert names[-1] == "Tier boundary: agents tier-3 imports"


def test_smoke_steps_with_sync_has_six_steps() -> None:
    assert len(smoke_steps(include_uv_sync=True)) == 6


def test_fast_push_steps_have_no_pytest() -> None:
    steps = fast_steps()
    assert len(steps) == 5
    names = " ".join(step.name for step in steps)
    assert "smoke unit tests" not in names.lower()
