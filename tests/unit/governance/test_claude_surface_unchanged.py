# © Artur Czarnecki. All rights reserved.
"""The Cursor software factory must not leak into the Claude Code surface.

Deliberately not a frozen hash: future Claude updates stay possible. It asserts isolation instead.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLAUDE = ROOT / ".claude"
FACTORY_TOKENS = ("igx_factory", "igx_factory_stop", "igx_factory_supervisor", "igx_test_lock", "igx-factory.mdc")


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


def _load_sync():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("sync_agent_env_iso", ROOT / "scripts/governance/sync_agent_env.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claude_settings_do_not_reference_factory_scripts() -> None:
    text = (CLAUDE / "settings.json").read_text(encoding="utf-8")
    for token in FACTORY_TOKENS[:4]:
        assert token not in text


def test_no_claude_file_mentions_the_factory() -> None:
    files = [p for p in CLAUDE.rglob("*") if p.is_file()] + [ROOT / "CLAUDE.md"]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in ("igx-factory.mdc", "igx_factory"):
            assert token not in text, f"{path.relative_to(ROOT)} mentions {token}"


def test_claude_stop_hook_is_still_the_stop_guard() -> None:
    settings = json.loads((CLAUDE / "settings.json").read_text(encoding="utf-8"))
    commands = [hook["command"] for group in settings["hooks"]["Stop"] for hook in group["hooks"]]
    assert any("igx_stop_guard.py" in command for command in commands)
    assert not any("factory" in command for command in commands)


def test_generated_claude_targets_equal_committed_files() -> None:
    expected = _load_sync().expected_files()
    claude_targets = {path: text for path, text in expected.items() if CLAUDE in path.parents}
    assert claude_targets
    for path, text in claude_targets.items():
        assert path.is_file(), f"missing {path.relative_to(ROOT)}"
        assert _norm(path.read_text(encoding="utf-8")) == _norm(text), path.relative_to(ROOT)


def test_cursor_stop_hook_is_not_the_claude_guard() -> None:
    hooks = json.loads((ROOT / ".cursor" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    for entry in hooks["stop"]:
        assert "igx_stop_guard.py" not in entry["command"]
