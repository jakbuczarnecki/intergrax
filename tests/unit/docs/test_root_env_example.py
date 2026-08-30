# © Artur Czarnecki. All rights reserved.

"""Bounded consistency checks for the root platform `.env.example` template."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from intergrax.runtime.config.forbidden_generation_model_env import (
    BOOTSTRAP_FORBIDDEN_GENERATION_MODEL_ENV_NAMES,
)
from intergrax.runtime.notifications.backend_contract import NotificationBackend

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
ENV_EXAMPLE = REPO / ".env.example"

_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
_COMMENTED_ASSIGNMENT = re.compile(r"^#\s*([A-Z][A-Z0-9_]*)=")
_SECRETISH = re.compile(
    r"(?i)(?:sk-|xox[baprs]-|ghp_|github_pat_|AKIA)[A-Za-z0-9_\-]{8,}"
)
_REMOVED_EMBEDDING_SELECTION = (
    "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL",
    "INTERGRAX_DEFAULT_VLLM_EMBED_MODEL",
    "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL",
    "INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL",
    "INTERGRAX_DEFAULT_HF_EMBED_MODEL",
    "INTERGRAX_RAG_EMBEDDING_PROVIDER",
)
_APP_PREFIXES = ("LEGAL_", "RESEARCH_", "LAB_")
_PROOF_PREFIXES = ("LKW_", "INTERGRAX_LKW_")
_SECRET_NAME = re.compile(r"(?:API_KEY|TOKEN|SECRET|PASSWORD|ROUTING_KEY)\s*$")


def _text() -> str:
    return ENV_EXAMPLE.read_text(encoding="utf-8")


def _active_assignments(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.match(line)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def _mentioned_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = _COMMENTED_ASSIGNMENT.match(line)
            if match:
                keys.add(match.group(1))
            continue
        match = _ASSIGNMENT.match(line)
        if match:
            keys.add(match.group(1))
    return keys


def test_canonical_llm_and_embedding_pairs_are_present() -> None:
    text = _text()
    assert "INTERGRAX_LLM_PROVIDER=ollama" in text
    assert "INTERGRAX_LLM_MODEL=llama3.1:latest" in text
    assert "INTERGRAX_EMBEDDING_PROVIDER=ollama" in text
    assert "INTERGRAX_EMBEDDING_MODEL=nomic-embed-text" in text
    assignments = dict(_active_assignments(text))
    for key in BOOTSTRAP_FORBIDDEN_GENERATION_MODEL_ENV_NAMES:
        assert key not in assignments


def test_removed_embedding_and_app_and_proof_keys_are_absent() -> None:
    text = _text()
    keys = _mentioned_keys(text)
    for name in _REMOVED_EMBEDDING_SELECTION:
        assert name not in text
    matching = [
        key for key in keys if key.startswith(_APP_PREFIXES + _PROOF_PREFIXES)
    ]
    assert matching == [], f"application/proof keys still present: {matching}"
    assert "GH_TOKEN" not in text
    assert "INTERGRAX_USE_WORKER_QUEUE" not in text
    assert "HF_TOKEN" not in text
    assert "VECTOR_STORE_ID" not in text


def test_no_duplicate_active_assignments() -> None:
    names = [name for name, _ in _active_assignments(_text())]
    dupes = [name for name, count in Counter(names).items() if count > 1]
    assert dupes == [], f"duplicate assignments: {dupes}"


def test_platform_configuration_reference_exists() -> None:
    text = _text()
    assert "docs/project/technical/guides/PLATFORM_CONFIGURATION.md" in text
    assert "practical example of common Intergrax platform configuration" in text


def test_notification_backend_comment_matches_runtime() -> None:
    text = _text()
    expected = {backend.value for backend in NotificationBackend}
    comment_lines = [
        line
        for line in text.splitlines()
        if "Allowed backends:" in line or "INTERGRAX_NOTIFICATION_BACKEND" in line
    ]
    joined = "\n".join(comment_lines)
    missing = sorted(value for value in expected if value not in joined)
    assert missing == [], f"notification backends missing from template comments: {missing}"


def test_export_journal_documents_enabled_by_default() -> None:
    text = _text()
    assert "INTERGRAX_EXPORT_JOURNAL" in text
    lowered = text.lower()
    assert "enabled by default" in lowered
    assert "intergrax_export_journal is enabled by default (1)" in lowered


def test_no_apparent_real_secrets() -> None:
    text = _text()
    assert _SECRETISH.search(text) is None
    for name, value in _active_assignments(text):
        stripped = value.strip().strip('"').strip("'")
        if not stripped:
            continue
        if _SECRET_NAME.search(name) or name.endswith("_KEY"):
            raise AssertionError(f"non-empty secret-like assignment: {name}")
