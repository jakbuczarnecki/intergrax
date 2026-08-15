# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-6D / CTX-UCL-CLOSEOUT-1: cross-domain legacy summary surface guard."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]

_PRODUCTION_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax" / "context",
    _REPO_ROOT / "intergrax" / "runtime",
    _REPO_ROOT / "intergrax" / "applications",
    _REPO_ROOT / "intergrax" / "agents",
    _REPO_ROOT / "intergrax" / "memory",
    _REPO_ROOT / "intergrax" / "prompts",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "agents",
)

_BANNED_TOKENS = (
    "HistorySummaryPromptBuilder",
    "DefaultHistorySummaryPromptBuilder",
    "HistorySummaryPromptBundle",
    "build_history_summary_prompt",
    "history_prompt_builder",
    "history_summary_cache",
    "conversation_summary_cache",
    "cached_history_summary",
)

_CANONICAL_OWNERSHIP: dict[str, Path] = {
    "resolve_ucl_context_plan": (
        _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "ucl_orchestration.py"
    ),
    "MessageSequenceArtifactExecutor": (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "token_optimization"
        / "message_sequence_artifact.py"
    ),
    "message_sequence_summarization.v1": (
        _REPO_ROOT / "intergrax" / "runtime" / "wiring" / "context_runtime_bridge.py"
    ),
    "INTERNAL_OPTIMIZATION_CALL": (
        _REPO_ROOT / "intergrax" / "runtime" / "context_lifecycle" / "contracts.py"
    ),
    "ArtifactCreationReservation": (
        _REPO_ROOT / "intergrax" / "runtime" / "context_lifecycle" / "contracts.py"
    ),
}

_SKIP_PATH_MARKERS = (
    "/docker/runtime-context/",
    "/vendor/",
    "/__pycache__/",
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
            if any(marker in relative for marker in _SKIP_PATH_MARKERS):
                continue
            files.append(path)
    return sorted(files)


def _module_defines_symbol(path: Path, symbol: str) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return True
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return True
    return symbol in source


@pytest.mark.parametrize("token", _BANNED_TOKENS)
def test_production_paths_forbid_legacy_history_summary_tokens(token: str) -> None:
    offenders: list[str] = []
    for path in _iter_production_python_files():
        source = path.read_text(encoding="utf-8")
        if token in source:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f"{token} found in: {offenders}"


@pytest.mark.parametrize("symbol,owner_path", list(_CANONICAL_OWNERSHIP.items()))
def test_canonical_ucl_symbols_remain_in_owning_modules(
    symbol: str,
    owner_path: Path,
) -> None:
    assert owner_path.is_file(), f"missing canonical owner module for {symbol}: {owner_path}"
    assert _module_defines_symbol(owner_path, symbol), (
        f"{symbol} must be defined or declared in {owner_path.relative_to(_REPO_ROOT)}"
    )


def test_production_scan_roots_cover_agents_and_prompts() -> None:
    relative_roots = {path.relative_to(_REPO_ROOT).as_posix() for path in _PRODUCTION_SCAN_ROOTS}
    assert "intergrax/agents" in relative_roots
    assert "intergrax/prompts" in relative_roots
