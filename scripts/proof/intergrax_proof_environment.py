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

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_DOTENV_FILENAME = ".env"


@dataclass(frozen=True, slots=True)
class ProofEnvironmentLoadResult:
    """Outcome of loading proof environment — paths only, no secret values."""

    dotenv_path: Path | None
    loaded: bool


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


def load_proof_environment(
    *,
    proof_package_dir: Path,
    repository_root: Path,
) -> ProofEnvironmentLoadResult:
    """Load the nearest proof ``.env`` without overwriting existing process variables."""
    dotenv_path = find_proof_dotenv(
        proof_package_dir=proof_package_dir,
        repository_root=repository_root,
    )
    if dotenv_path is None:
        return ProofEnvironmentLoadResult(dotenv_path=None, loaded=False)
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return ProofEnvironmentLoadResult(dotenv_path=dotenv_path, loaded=True)
