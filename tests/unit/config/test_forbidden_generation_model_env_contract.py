# © Artur Czarnecki. All rights reserved.

"""CONFIG-6 — regression guard against legacy generation model-selection env names."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_GENERATION_MODEL_ENV_NAMES = frozenset(
    {
        "INTERGRAX_DEFAULT_OLLAMA_MODEL",
        "INTERGRAX_DEFAULT_VLLM_MODEL",
        "INTERGRAX_DEFAULT_LLAMA_CPP_MODEL",
        "INTERGRAX_DEFAULT_OPENAI_MODEL",
        "INTERGRAX_DEFAULT_CLAUDE_MODEL",
        "INTERGRAX_DEFAULT_GEMINI_MODEL",
        "INTERGRAX_DEFAULT_MISTRAL_MODEL",
        "INTERGRAX_DEFAULT_GROQ_MODEL",
        "INTERGRAX_DEFAULT_TOGETHER_MODEL",
        "INTERGRAX_DEFAULT_FIREWORKS_MODEL",
        "INTERGRAX_DEFAULT_OPENROUTER_MODEL",
        "INTERGRAX_DEFAULT_DEEPSEEK_MODEL",
        "INTERGRAX_DEFAULT_XAI_MODEL",
        "INTERGRAX_DEFAULT_COHERE_MODEL",
        "INTERGRAX_DEFAULT_COHERE_NATIVE_MODEL",
        "INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_MODEL",
        "INTERGRAX_DEFAULT_VERTEX_GEMINI_MODEL",
        "INTERGRAX_DEFAULT_BEDROCK_MODEL_ID",
    }
)

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

_THIS_FILE = Path(__file__).resolve()


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
            if path.resolve() == _THIS_FILE:
                continue
            files.append(path)
    return files


def _allowed_occurrences(text: str, path: Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    if rel == "tests/unit/config/test_forbidden_generation_model_env_contract.py":
        return True
    if "FORBIDDEN_GENERATION_MODEL_ENV_NAMES" in text:
        return True
    if rel.startswith("tests/") or "/tests/" in rel:
        return True
    return False


@pytest.mark.parametrize("env_name", sorted(FORBIDDEN_GENERATION_MODEL_ENV_NAMES))
def test_forbidden_generation_model_env_not_present_in_active_repo(env_name: str) -> None:
    pattern = re.compile(re.escape(env_name))
    violations: list[str] = []
    for path in _iter_scan_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if env_name not in text:
            continue
        if _allowed_occurrences(text, path):
            continue
        violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert not violations, f"{env_name} found in: {', '.join(violations)}"
