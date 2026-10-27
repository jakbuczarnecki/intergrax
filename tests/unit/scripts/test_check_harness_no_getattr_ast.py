# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "maintenance"))

from check_harness_no_getattr import _scan_source  # noqa: E402


def test_string_literal_getattr_not_flagged() -> None:
    source = 'def ok() -> None:\n    assert "getattr(" not in text\n'
    assert _scan_source("tests/unit/example.py", source) == []


def test_real_getattr_call_flagged() -> None:
    source = "def bad() -> None:\n    getattr(x, 'y')\n"
    hits = _scan_source("intergrax/example.py", source)
    assert len(hits) == 1
    assert "getattr" in hits[0]
