# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SYNC = ROOT / "scripts" / "governance" / "sync_agent_env.py"


def test_agent_env_adapters_match_ssot() -> None:
    completed = subprocess.run(
        [sys.executable, str(SYNC), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_every_rule_has_loading_mode_and_description() -> None:
    sys.path.insert(0, str(SYNC.parent))
    try:
        import sync_agent_env  # noqa: WPS433
    finally:
        sys.path.remove(str(SYNC.parent))
    rules = sync_agent_env.load_rules()
    assert rules["intergrax-core"]["loading"] == "always"
    assert [rid for rid, meta in rules.items() if meta["loading"] == "always"] == ["intergrax-core"]
    assert sync_agent_env.size_problems() == []
    assert all(meta["loading"] in sync_agent_env.LOADING_MODES for meta in rules.values())
