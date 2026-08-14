# © Artur Czarnecki. All rights reserved.

"""CONFIG-6 — regression guard against legacy generation model-selection env names."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from intergrax.runtime.config.forbidden_generation_model_env import (
    FORBIDDEN_GENERATION_MODEL_ENV_NAMES,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "infra",
    _REPO_ROOT / "docs",
    _REPO_ROOT / ".github",
)

_SKIP_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "build",
    "dist",
    "runtime-context",
}

_ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".sh",
    ".bat",
    ".ps1",
    ".yml",
    ".yaml",
    ".env",
    ".example",
}

_REGISTRY_FILE = (
    _REPO_ROOT / "intergrax" / "runtime" / "config" / "forbidden_generation_model_env.py"
)


def _iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_PARTS for part in path.parts):
                continue
            if path.suffix not in _ALLOWED_SUFFIXES and path.name != ".env.example":
                continue
            if path.resolve() == _REGISTRY_FILE.resolve():
                continue
            files.append(path)
    return files


def _allowed_occurrences(text: str, path: Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    if "FORBIDDEN_GENERATION_MODEL_ENV_NAMES" in text:
        return True
    if rel.startswith("tests/") or "/tests/" in rel:
        return True
    return False


@pytest.mark.parametrize("env_name", sorted(FORBIDDEN_GENERATION_MODEL_ENV_NAMES))
def test_forbidden_generation_model_env_not_present_in_active_repo(env_name: str) -> None:
    violations: list[str] = []
    for path in _iter_scan_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if env_name not in text:
            continue
        if _allowed_occurrences(text, path):
            continue
        violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert not violations, f"{env_name} found in: {', '.join(violations)}"
