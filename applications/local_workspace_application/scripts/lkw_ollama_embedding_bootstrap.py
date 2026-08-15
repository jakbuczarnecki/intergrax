# © Artur Czarnecki. All rights reserved.

"""Shared LKW embedding profile resolution and Ollama model provisioning."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

EMBEDDING_PROFILE_RESOLUTION_CODE = (
    "from intergrax.rag.embedding.registry.profile import embedding_profile_from_env; "
    "profile = embedding_profile_from_env(); "
    "print(profile.provider); "
    "print(profile.model or '')"
)
OLLAMA_DEFAULT_MODEL_RESOLUTION_CODE = (
    "from intergrax.rag.embedding.providers.ollama_embedding_provider "
    "import OllamaEmbeddingProvider; "
    "print(OllamaEmbeddingProvider.DEFAULT_MODEL)"
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


def resolve_embedding_profile_from_container(
    *,
    compose_exec_args: Callable[..., list[str]],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    cwd: Path,
    timeout_seconds: int,
    run_command_kwargs: Mapping[str, object] | None = None,
) -> tuple[str, str]:
    extra_kwargs = dict(run_command_kwargs or {})
    completed = run_command(
        compose_exec_args(
            "exec",
            "-T",
            "local_workspace",
            "python",
            "-c",
            EMBEDDING_PROFILE_RESOLUTION_CODE,
        ),
        cwd=cwd,
        timeout=timeout_seconds,
        **extra_kwargs,
    )
    if completed.returncode != 0:
        raise OllamaEmbeddingBootstrapError("embedding_profile_resolution_failed")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise OllamaEmbeddingBootstrapError("embedding_profile_resolution_failed")
    provider = lines[0].strip().lower()
    model = lines[1].strip()
    return provider, model


def resolve_ollama_default_model_from_container(
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
            OLLAMA_DEFAULT_MODEL_RESOLUTION_CODE,
        ),
        cwd=cwd,
        timeout=timeout_seconds,
        **extra_kwargs,
    )
    if completed.returncode != 0:
        raise OllamaEmbeddingBootstrapError("embedding_model_resolution_failed")
    return validate_resolved_embedding_model(completed.stdout)


def resolve_ollama_embedding_model(
    *,
    compose_exec_args: Callable[..., list[str]],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    cwd: Path,
    timeout_seconds: int,
    run_command_kwargs: Mapping[str, object] | None = None,
) -> str:
    provider, model = resolve_embedding_profile_from_container(
        compose_exec_args=compose_exec_args,
        run_command=run_command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        run_command_kwargs=run_command_kwargs,
    )
    if provider != "ollama":
        raise OllamaEmbeddingBootstrapError("embedding_provider_not_ollama")
    if model:
        return validate_resolved_embedding_model(model)
    return resolve_ollama_default_model_from_container(
        compose_exec_args=compose_exec_args,
        run_command=run_command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        run_command_kwargs=run_command_kwargs,
    )


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


def ensure_ollama_embedding_model_if_configured(
    *,
    compose_exec_args: Callable[..., list[str]],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    cwd: Path,
    timeout_seconds: int,
    run_command_kwargs: Mapping[str, object] | None = None,
) -> str | None:
    provider, model = resolve_embedding_profile_from_container(
        compose_exec_args=compose_exec_args,
        run_command=run_command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        run_command_kwargs=run_command_kwargs,
    )
    if provider != "ollama":
        return None
    resolved_model = (
        validate_resolved_embedding_model(model)
        if model
        else resolve_ollama_default_model_from_container(
            compose_exec_args=compose_exec_args,
            run_command=run_command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            run_command_kwargs=run_command_kwargs,
        )
    )
    ensure_ollama_embedding_model(
        resolved_model,
        compose_exec_args=compose_exec_args,
        run_command=run_command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        run_command_kwargs=run_command_kwargs,
    )
    return resolved_model
