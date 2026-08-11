# © Artur Czarnecki. All rights reserved.

"""Shared LKW local Ollama embedding model resolution and provisioning."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

MODEL_RESOLUTION_CODE = (
    "import os; "
    "from intergrax.rag.embedding.providers.ollama_embedding_provider "
    "import OllamaEmbeddingProvider; "
    "print(os.getenv(OllamaEmbeddingProvider.ENV_MODEL) "
    "or OllamaEmbeddingProvider.DEFAULT_MODEL)"
)
MAX_EMBEDDING_MODEL_LENGTH = 256


class OllamaEmbeddingBootstrapError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_resolved_embedding_model(output: str) -> str:
    if not isinstance(output, str):
        raise OllamaEmbeddingBootstrapError("embedding_model_resolution_failed")
    non_empty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(non_empty_lines) != 1:
        raise OllamaEmbeddingBootstrapError("embedding_model_resolution_failed")
    model_name = non_empty_lines[0]
    if (
        not model_name
        or len(model_name) > MAX_EMBEDDING_MODEL_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in model_name)
        or any(character.isspace() for character in model_name)
    ):
        raise OllamaEmbeddingBootstrapError("embedding_model_resolution_failed")
    return model_name


def resolve_ollama_embedding_model(
    *,
    compose_exec_args: Callable[..., list[str]],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    cwd: Path,
    timeout_seconds: int,
    run_command_kwargs: Mapping[str, object] | None = None,
) -> str:
    extra_kwargs = dict(run_command_kwargs or {})
    completed = run_command(
        compose_exec_args(
            "exec",
            "-T",
            "local_workspace",
            "python",
            "-c",
            MODEL_RESOLUTION_CODE,
        ),
        cwd=cwd,
        timeout=timeout_seconds,
        **extra_kwargs,
    )
    if completed.returncode != 0:
        raise OllamaEmbeddingBootstrapError("embedding_model_resolution_failed")
    return validate_resolved_embedding_model(completed.stdout)


def ensure_ollama_embedding_model(
    model_name: str,
    *,
    compose_exec_args: Callable[..., list[str]],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    cwd: Path,
    timeout_seconds: int,
    run_command_kwargs: Mapping[str, object] | None = None,
) -> None:
    extra_kwargs = dict(run_command_kwargs or {})
    completed = run_command(
        compose_exec_args(
            "exec",
            "-T",
            "ollama",
            "ollama",
            "pull",
            model_name,
        ),
        cwd=cwd,
        timeout=timeout_seconds,
        **extra_kwargs,
    )
    if completed.returncode != 0:
        raise OllamaEmbeddingBootstrapError("embedding_model_pull_failed")
