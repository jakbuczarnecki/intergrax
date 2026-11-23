from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MOD = ROOT / "scripts/hooks/cursor_development_push.py"


def _load():
    spec = importlib.util.spec_from_file_location("cursor_development_push_test", MOD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_push_refspec_matrix() -> None:
    m = _load()
    assert m.push_target_is_allowed_development("origin development")
    assert not m.push_target_is_allowed_development("origin main")
    assert not m.push_target_is_allowed_development("upstream development")
    assert not m.push_target_is_allowed_development("origin development --force")
