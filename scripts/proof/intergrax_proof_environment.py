# © Artur Czarnecki. All rights reserved.

"""Canonical proof-package environment loader (PROOF-ENV-CONTRACT-1).

Contract::

    process environment
    > nearest .env found by walking upward from the proof package directory
    > safe defaults

Only one ``.env`` file is loaded. Search stops at repository root.
Process environment is never overwritten (``override=False``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

_DOTENV_FILENAME = ".env"


@dataclass(frozen=True, slots=True)
class ProofEnvironmentLoadResult:
    """Outcome of loading proof environment — paths only, no secret values."""

    dotenv_path: Path | None
    loaded: bool


@dataclass(frozen=True, slots=True)
class ResolvedProofEnvironment:
    """Isolated proof environment without mutating ``os.environ``."""

    environment: dict[str, str]
    dotenv_path: Path | None
    dotenv_loaded: bool


def _resolve_proof_boundary(
    *,
    proof_package_dir: Path,
    repository_root: Path,
) -> tuple[Path, Path]:
    proof_dir = proof_package_dir.resolve()
    repo_root = repository_root.resolve()
    if proof_dir != repo_root:
        try:
            proof_dir.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                "proof_package_dir must be repository_root or a descendant of "
                "repository_root"
            ) from exc
    return proof_dir, repo_root


def find_proof_dotenv(
    *,
    proof_package_dir: Path,
    repository_root: Path,
) -> Path | None:
    """Return the nearest ``.env`` on the path from proof package to repository root."""
    proof_dir, repo_root = _resolve_proof_boundary(
        proof_package_dir=proof_package_dir,
        repository_root=repository_root,
    )
    current = proof_dir
    while True:
        candidate = current / _DOTENV_FILENAME
        if candidate.is_file():
            return candidate
        if current == repo_root:
            return None
        current = current.parent


def _proof_environment_defaults() -> dict[str, str]:
    return {}


def resolve_proof_environment(
    *,
    proof_package_dir: Path,
    repository_root: Path,
    base_environment: Mapping[str, str] | None = None,
    defaults: Mapping[str, str] | None = None,
) -> ResolvedProofEnvironment:
    """Build an isolated proof environment without mutating ``os.environ``.

    Precedence: ``base_environment`` (process/operator) > nearest proof ``.env`` >
    ``defaults``.
    """
    base = dict(base_environment if base_environment is not None else os.environ)
    effective_defaults = dict(
        defaults if defaults is not None else _proof_environment_defaults()
    )
    dotenv_path = find_proof_dotenv(
        proof_package_dir=proof_package_dir,
        repository_root=repository_root,
    )

    merged: dict[str, str] = dict(effective_defaults)
    dotenv_loaded = False
    if dotenv_path is not None:
        for key, value in dotenv_values(dotenv_path).items():
            if value is not None:
                merged[key] = value
        dotenv_loaded = True

    merged.update(base)
    return ResolvedProofEnvironment(
        environment=merged,
        dotenv_path=dotenv_path,
        dotenv_loaded=dotenv_loaded,
    )


def _apply_resolved_environment_to_process(
    resolved: ResolvedProofEnvironment,
) -> None:
    for key, value in resolved.environment.items():
        if key not in os.environ:
            os.environ[key] = value


def load_proof_environment(
    *,
    proof_package_dir: Path,
    repository_root: Path,
) -> ProofEnvironmentLoadResult:
    """Load the nearest proof ``.env`` without overwriting existing process variables."""
    resolved = resolve_proof_environment(
        proof_package_dir=proof_package_dir,
        repository_root=repository_root,
        base_environment=os.environ,
    )
    _apply_resolved_environment_to_process(resolved)
    return ProofEnvironmentLoadResult(
        dotenv_path=resolved.dotenv_path,
        loaded=resolved.dotenv_loaded,
    )


def bootstrap_process_environment(
    *,
    proof_package_dir: Path,
    repository_root: Path,
) -> ProofEnvironmentLoadResult:
    """Load canonical proof environment into the running process before profile resolution.

    Shared entry point for proof runners and qualification harnesses. Safe to call
    multiple times: ``override=False`` preserves variables already set in the process.
    """
    return load_proof_environment(
        proof_package_dir=proof_package_dir,
        repository_root=repository_root,
    )
