# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-6D: production-path guard against legacy history-summary surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]

_PRODUCTION_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime",
    _REPO_ROOT / "intergrax" / "applications",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "agents",
)

_BANNED_TOKENS = (
    "HistorySummaryPromptBuilder",
    "DefaultHistorySummaryPromptBuilder",
    "HistorySummaryPromptBundle",
    "build_history_summary_prompt",
    "history_prompt_builder",
    "summary_cache",
    "history_summary_cache",
    "conversation_summary_cache",
)

_ALLOWED_TOKENS = (
    "MessageSequenceArtifactExecutor",
    "message_sequence_summarization.v1",
    "INTERNAL_OPTIMIZATION_CALL",
)


def _iter_production_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if "/docker/runtime-context/" in relative:
                continue
            files.append(path)
    return sorted(files)


@pytest.mark.parametrize("token", _BANNED_TOKENS)
def test_production_paths_forbid_legacy_history_summary_tokens(token: str) -> None:
    offenders: list[str] = []
    for path in _iter_production_python_files():
        source = path.read_text(encoding="utf-8")
        if token in source:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f"{token} found in: {offenders}"


def test_production_paths_preserve_ucl_summary_creation_tokens() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _iter_production_python_files()
    )
    for token in _ALLOWED_TOKENS:
        assert token in combined, f"expected canonical UCL token present: {token}"
