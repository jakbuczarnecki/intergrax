# © Artur Czarnecki. All rights reserved.

"""Application-owned live proof evidence location and references."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OLD_PROOF = REPO / "docs" / "proof" / "slack_conversation_runtime_live_proof.md"
NEW_PROOF = (
    REPO
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "proof"
    / "SLACK_CONVERSATION_RUNTIME_LIVE_PROOF.md"
)
PROOF_INDEX = NEW_PROOF.parent / "README.md"
LKW_PLAN = (
    REPO / "applications" / "local_workspace_application" / "docs" / "IMPLEMENTATION_PLAN.md"
)
LKW_DISCOVERY = (
    REPO / "applications" / "local_workspace_application" / "docs" / "SLACK_MVP_DISCOVERY.md"
)
SLACK_USAGE = (
    REPO
    / "intergrax"
    / "integrations"
    / "providers"
    / "conversation_channel"
    / "slack"
    / "USAGE.md"
)
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_OLD_PATH = "/".join(("docs", "proof", "slack_conversation_runtime_live_proof.md"))
_SECRET_RE = re.compile(
    r"(?:xapp-[A-Za-z0-9]{8,}|xoxb-[A-Za-z0-9-]{8,}|"
    r"Authorization:\s*Bearer\s+\S+|"
    r"INTERGRAX_SLACK_(?:APP|BOT)_TOKEN=\S+)"
)
_THIS_TEST = Path(__file__).resolve()


def _resolve_md_targets(source: Path, needle: str) -> list[Path]:
    text = source.read_text(encoding="utf-8")
    resolved: list[Path] = []
    for raw in _MD_LINK.findall(text):
        if needle not in raw:
            continue
        target = (source.parent / raw).resolve()
        resolved.append(target)
    return resolved


def test_slack_live_proof_owned_by_lkw_application() -> None:
    assert NEW_PROOF.is_file()
    assert not OLD_PROOF.exists()

    text = NEW_PROOF.read_text(encoding="utf-8")
    assert "evidence owner: Local Knowledge Workspace" in text
    assert "verified platform capability: Slack conversation-channel runtime" in text
    assert "reference host: applications/local_workspace_application" in text
    assert "verified runtime commit: f6daddf7bdb32beb848266775011940c7c471f86" in text
    assert "certification commit: 30e0341c80a32c9e0d1c5d7adde046ad99d8b6c4" in text
    assert "SUMMARY=PASS" in text
    assert "proof harness: scripts/proof/slack_conversation_channel_live_proof.py" in text
    assert "is not application-owned" in text
    assert _SECRET_RE.search(text) is None

    for source in (LKW_PLAN, LKW_DISCOVERY):
        targets = _resolve_md_targets(source, "SLACK_CONVERSATION_RUNTIME_LIVE_PROOF.md")
        assert targets, f"missing proof link: {source}"
        for target in targets:
            assert target.is_file(), f"broken proof link: {source} -> {target}"
            assert target == NEW_PROOF.resolve()

    usage = SLACK_USAGE.read_text(encoding="utf-8")
    for forbidden in (
        "applications/",
        "agents/",
        "Local Knowledge Workspace",
        "SLACK_CONVERSATION_RUNTIME_LIVE_PROOF.md",
    ):
        assert forbidden not in usage, f"platform USAGE must not reference {forbidden!r}"
    for required in (
        "Socket Mode",
        "Web API",
        "MESSAGE",
        "ACTION",
        "runtime_binding_supported",
    ):
        assert required in usage
    assert _SECRET_RE.search(usage) is None

    index = PROOF_INDEX.read_text(encoding="utf-8")
    index_targets = _resolve_md_targets(PROOF_INDEX, "SLACK_CONVERSATION_RUNTIME_LIVE_PROOF.md")
    assert index_targets
    assert all(t == NEW_PROOF.resolve() for t in index_targets)
    assert "Local Knowledge Workspace" in index

    tracked = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        text=False,
    ).split(b"\0")
    stale: list[str] = []
    for raw in tracked:
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace")
        path = REPO / rel
        if path.resolve() == _THIS_TEST:
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf"}:
            continue
        if path.name in {"uv.lock", "Cargo.lock"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _OLD_PATH in content:
            stale.append(rel)
    assert not stale, f"stale old-path references: {stale}"
