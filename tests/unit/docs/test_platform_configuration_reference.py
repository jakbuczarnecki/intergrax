# © Artur Czarnecki. All rights reserved.

"""Bounded consistency checks for the canonical platform configuration reference."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / "docs" / "project" / "technical" / "guides" / "PLATFORM_CONFIGURATION.md"
MAP = REPO / "docs" / "project" / "technical" / "DOCUMENTATION_MAP.md"
GUIDES_INDEX = REPO / "docs" / "project" / "technical" / "guides" / "README.md"

_REMOVED_EMBEDDING_SELECTION = (
    "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL",
    "INTERGRAX_DEFAULT_VLLM_EMBED_MODEL",
    "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL",
    "INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL",
    "INTERGRAX_DEFAULT_HF_EMBED_MODEL",
    "INTERGRAX_RAG_EMBEDDING_PROVIDER",
)

_APP_PREFIXES = (
    "LOCAL_WORKSPACE_",
    "LEGAL_",
    "RESEARCH_",
    "LAB_",
)

_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")
_SECRETISH = re.compile(
    r"(?i)(?:sk-|xox[baprs]-|ghp_|github_pat_|AKIA)[A-Za-z0-9_\-]{8,}"
)


def _quick_reference_block(text: str) -> str:
    start = text.index("## Quick reference")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def test_canonical_llm_and_embedding_pairs_are_documented() -> None:
    text = DOC.read_text(encoding="utf-8")
    for heading in (
        "### INTERGRAX_LLM_PROVIDER",
        "### INTERGRAX_LLM_MODEL",
        "### INTERGRAX_EMBEDDING_PROVIDER",
        "### INTERGRAX_EMBEDDING_MODEL",
    ):
        assert heading in text, f"missing heading {heading}"


def test_removed_embedding_vars_are_not_current_configuration() -> None:
    text = DOC.read_text(encoding="utf-8")
    quick = _quick_reference_block(text)
    for name in _REMOVED_EMBEDDING_SELECTION:
        assert f"`{name}`" not in quick, f"{name} listed in quick reference"
        assert name in text, f"{name} should be named as not current"
    assert "not current supported model-selection" in text.replace("**", "")


def test_application_prefixes_are_not_in_platform_option_table() -> None:
    quick = _quick_reference_block(DOC.read_text(encoding="utf-8"))
    for prefix in _APP_PREFIXES:
        assert prefix not in quick, f"application prefix {prefix} in quick reference"


def test_markdown_links_resolve() -> None:
    text = DOC.read_text(encoding="utf-8")
    doc_dir = DOC.parent
    missing: list[str] = []
    for _, target in _MD_LINK.findall(text):
        href = target.strip()
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        path_part = href.split("#", 1)[0].strip()
        if not path_part:
            continue
        resolved = (doc_dir / path_part).resolve()
        if not resolved.exists():
            missing.append(href)
    assert not missing, "broken relative links: " + ", ".join(missing)


def test_no_secrets_and_headings_are_valid() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert DOC.is_file()
    assert text.startswith("# Intergrax Platform Configuration\n")
    assert _SECRETISH.search(text) is None
    fences = text.count("```")
    assert fences % 2 == 0, "unbalanced markdown fences"


def test_canonical_doc_is_linked_from_technical_indexes() -> None:
    map_text = MAP.read_text(encoding="utf-8")
    guides_text = GUIDES_INDEX.read_text(encoding="utf-8")
    assert "PLATFORM_CONFIGURATION.md" in map_text
    assert "PLATFORM_CONFIGURATION.md" in guides_text
    assert "guides/PLATFORM_CONFIGURATION.md" in map_text
