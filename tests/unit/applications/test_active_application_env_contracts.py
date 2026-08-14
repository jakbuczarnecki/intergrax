# © Artur Czarnecki. All rights reserved.

"""CONFIG-5 — bounded consistency of active application env templates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APPLICATIONS = _REPO_ROOT / "applications"

_ACTIVE_APPLICATIONS = (
    "local_workspace_application",
    "legal_application",
    "research_application",
    "lab_application",
    "poc_template_application",
    "intergrax_assistant_application",
    "governed_contractor_application",
    "dispute_sim_application",
)

_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Z][A-Z0-9_]*)=", re.MULTILINE)
_COMMENTED_ASSIGNMENT = re.compile(r"^#\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)

_LEGACY_EMBEDDING_KEYS = (
    "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL",
    "INTERGRAX_DEFAULT_VLLM_EMBED_MODEL",
    "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL",
    "INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL",
    "INTERGRAX_DEFAULT_HF_EMBED_MODEL",
    "INTERGRAX_RAG_EMBEDDING_PROVIDER",
)

_REDUNDANT_GENERATION_DEFAULTS = (
    "INTERGRAX_DEFAULT_OLLAMA_MODEL",
    "INTERGRAX_DEFAULT_VLLM_MODEL",
    "INTERGRAX_DEFAULT_LLAMA_CPP_MODEL",
)


def _named_keys(text: str) -> set[str]:
    return set(_ASSIGNMENT.findall(text)) | set(_COMMENTED_ASSIGNMENT.findall(text))


def _active_keys(text: str) -> set[str]:
    return set(_ASSIGNMENT.findall(text))


def _env_example_text(app: str) -> str:
    path = _APPLICATIONS / app / ".env.example"
    assert path.is_file(), f"missing {path.as_posix()}"
    return path.read_text(encoding="utf-8")


def _compose_texts(app: str) -> list[tuple[Path, str]]:
    docker_dir = _APPLICATIONS / app / "docker"
    if not docker_dir.is_dir():
        return []
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(docker_dir.glob("docker-compose*.yml"))
    ]


@pytest.mark.parametrize("app", _ACTIVE_APPLICATIONS)
def test_active_env_example_uses_canonical_model_pairs(app: str) -> None:
    text = _env_example_text(app)
    named = _named_keys(text)
    active = _active_keys(text)

    for key in _LEGACY_EMBEDDING_KEYS:
        assert key not in named, f"{app}: legacy embedding key {key}"

    if "INTERGRAX_LLM_MODEL" in active:
        assert "INTERGRAX_LLM_PROVIDER" in active, f"{app}: missing INTERGRAX_LLM_PROVIDER"
        for key in _REDUNDANT_GENERATION_DEFAULTS:
            assert key not in named, f"{app}: redundant {key} with INTERGRAX_LLM_MODEL"

    if "INTERGRAX_EMBEDDING_MODEL" in active:
        assert "INTERGRAX_EMBEDDING_PROVIDER" in active, (
            f"{app}: missing INTERGRAX_EMBEDDING_PROVIDER"
        )


@pytest.mark.parametrize("app", _ACTIVE_APPLICATIONS)
def test_active_compose_does_not_reintroduce_removed_model_keys(app: str) -> None:
    for path, text in _compose_texts(app):
        named = _named_keys(text.replace(": ", "="))
        for key in _LEGACY_EMBEDDING_KEYS:
            assert key not in text, f"{path.as_posix()}: legacy embedding key {key}"
        if "INTERGRAX_LLM_MODEL" in named or "INTERGRAX_LLM_MODEL:" in text:
            for key in _REDUNDANT_GENERATION_DEFAULTS:
                assert key not in text, f"{path.as_posix()}: redundant {key}"


def test_lkw_env_example_keeps_canonical_pairs() -> None:
    text = _env_example_text("local_workspace_application")
    assert "INTERGRAX_LLM_PROVIDER=" in text
    assert "INTERGRAX_LLM_MODEL=" in text
    assert "INTERGRAX_EMBEDDING_PROVIDER=ollama" in text
    assert "INTERGRAX_EMBEDDING_MODEL=nomic-embed-text" in text
    assert "INTERGRAX_DEFAULT_OLLAMA_MODEL" not in text
    assert "INTERGRAX_DEFAULT_VLLM_MODEL" not in text
    assert "PLATFORM_CONFIGURATION.md" in text


def test_lab_and_product_templates_point_at_platform_configuration() -> None:
    for app in (
        "lab_application",
        "legal_application",
        "research_application",
        "poc_template_application",
        "intergrax_assistant_application",
    ):
        text = _env_example_text(app)
        assert "PLATFORM_CONFIGURATION.md" in text, f"{app}: missing platform config pointer"
