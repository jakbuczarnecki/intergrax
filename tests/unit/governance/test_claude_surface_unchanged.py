# © Artur Czarnecki. All rights reserved.
"""The Cursor software factory must not leak into the Claude Code surface.

Two layers: isolation checks (no factory token in Claude files, Claude Stop hook intact, generated Claude files equal
the committed ones) and pinned git object hashes of the Claude autonomy surface.

Updating a pin: only when the Claude autonomy surface is changed on purpose. Run
`git rev-parse HEAD:<path>` after committing the change, replace the hash below in the same PR and name the change in
that PR's ticket. A pin failing without such a change means a Cursor-side edit leaked into the Claude surface.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CLAUDE = ROOT / ".claude"
FACTORY_TOKENS = ("igx_factory", "igx_factory_stop", "igx_factory_supervisor", "igx_test_lock", "igx-factory.mdc")

# git object ids (blob for a file, tree for a directory) recorded at the factory base c98d9ac0b, verified equal there.
CLAUDE_SURFACE_PINS = {
    "scripts/hooks/igx_stop_guard.py": "04f8ececb86a3f7a015dd4f2676a255c4f6e1d30",
    "governance/agent-control-plane": "e2719a9ad39e7c76fa220a66fda9f8ea4a109cfc",
    "governance/TRUSTED_POLICY_PIN.yaml": "e8d5614fa24b7166a63c05d48da3da2ea6ca266c",
    "agent-env/skills/igx-run-plan": "133a218aa44f0489c84bcac3b4199a709bf2cb5d",
    "agent-env/skills/igx-autonomy": "c0950b5155eb8953fd5bf7b10403029acfb15678",
    "agent-env/rules/intergrax-core.md": "78ecc6862d6ea4f15959fb1530b510f1d272bcb6",
    "agent-env/agents": "6b53d890d3506b5326bcc071708cfafe1f8f76b3",
    ".claude/settings.json": "77cac1626ad88f85095a91e5a8abab19f5292641",
}


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)  # noqa: S603, S607


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


def test_claude_autonomy_surface_matches_pinned_git_hashes() -> None:
    if _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not a git checkout")
    for path, pin in CLAUDE_SURFACE_PINS.items():
        committed = _git("rev-parse", f"HEAD:{path}")
        assert committed.returncode == 0, f"{path} is missing from HEAD: {committed.stderr.strip()}"
        assert committed.stdout.strip() == pin, f"{path} differs from its pin (see the update procedure in the header)"
        dirty = _git("status", "--porcelain", "--", path).stdout.strip()
        assert not dirty, f"{path} has uncommitted changes: {dirty}"


def test_pins_cover_every_claude_autonomy_entry_point() -> None:
    assert "scripts/hooks/igx_stop_guard.py" in CLAUDE_SURFACE_PINS
    assert any(path.startswith("governance/agent-control-plane") for path in CLAUDE_SURFACE_PINS)
    assert ".claude/settings.json" in CLAUDE_SURFACE_PINS
    assert all(len(pin) == 40 for pin in CLAUDE_SURFACE_PINS.values())
