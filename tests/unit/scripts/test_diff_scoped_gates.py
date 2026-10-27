# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "ci"))

from diff_scoped_gates import DEFAULT_RULES, run_rules  # noqa: E402
from git_changed_paths import paths_touch_prefix  # noqa: E402


def test_paths_touch_prefix() -> None:
    paths = frozenset({"agents/lab/contract.py", "docs/readme.md"})
    assert paths_touch_prefix(paths, ("agents/",)) is True
    assert paths_touch_prefix(paths, ("intergrax/",)) is False


def test_run_rules_skips_when_no_scope_hit() -> None:
    assert run_rules(frozenset({"README.md"}), DEFAULT_RULES) == 0
