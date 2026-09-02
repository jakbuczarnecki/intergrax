# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q3 proof environment loader regression (canonical .env precedence)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.proof.intergrax_proof_environment import load_proof_environment

_Q3_PROOF_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _Q3_PROOF_PACKAGE_DIR.parents[2]
_TAVILY_KEY = "INTERGRAX_TAVILY_API_KEY"


def _write_env(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_q3_proof_environment_loads_tavily_key_from_root_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    q3_dir = repo_root / "tests" / "system" / "functional_diagnostics_q3"
    q3_dir.mkdir(parents=True)
    _write_env(repo_root / ".env", f"{_TAVILY_KEY}=dotenv-tavily-key\n")
    monkeypatch.delenv(_TAVILY_KEY, raising=False)

    result = load_proof_environment(
        proof_package_dir=q3_dir,
        repository_root=repo_root,
    )

    assert result.loaded is True
    assert os.environ[_TAVILY_KEY] == "dotenv-tavily-key"


def test_q3_proof_environment_process_env_wins_over_root_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    q3_dir = repo_root / "tests" / "system" / "functional_diagnostics_q3"
    q3_dir.mkdir(parents=True)
    _write_env(repo_root / ".env", f"{_TAVILY_KEY}=dotenv-value-b\n")
    monkeypatch.setenv(_TAVILY_KEY, "process-value-a")

    result = load_proof_environment(
        proof_package_dir=q3_dir,
        repository_root=repo_root,
    )

    assert result.loaded is True
    assert os.environ[_TAVILY_KEY] == "process-value-a"


def test_q3_proof_package_resolves_repository_root_dotenv() -> None:
    root_env = _REPO_ROOT / ".env"
    if not root_env.is_file():
        pytest.skip("repository root .env not present in this checkout")

    dotenv_path = load_proof_environment(
        proof_package_dir=_Q3_PROOF_PACKAGE_DIR,
        repository_root=_REPO_ROOT,
    ).dotenv_path

    assert dotenv_path == root_env.resolve()
