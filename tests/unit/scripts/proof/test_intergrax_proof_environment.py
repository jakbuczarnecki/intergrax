# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.proof.intergrax_proof_environment import (
    find_proof_dotenv,
    load_proof_environment,
)


def _proof_package(repo_root: Path) -> Path:
    return repo_root / "platform_proofs" / "scenarios" / "foo"


def _write_env(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_proof_local_dotenv_wins(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    proof_env = _write_env(proof_dir / ".env", "PROOF_ONLY=1\n")
    _write_env(repo_root / ".env", "ROOT_ONLY=1\n")

    assert (
        find_proof_dotenv(
            proof_package_dir=proof_dir,
            repository_root=repo_root,
        )
        == proof_env
    )


def test_repository_root_fallback(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    root_env = _write_env(repo_root / ".env", "ROOT_ONLY=1\n")

    assert (
        find_proof_dotenv(
            proof_package_dir=proof_dir,
            repository_root=repo_root,
        )
        == root_env
    )


def test_intermediate_directory_dotenv_wins(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    platform_env = _write_env(
        repo_root / "platform_proofs" / ".env",
        "PLATFORM_ONLY=1\n",
    )
    _write_env(repo_root / ".env", "ROOT_ONLY=1\n")

    assert (
        find_proof_dotenv(
            proof_package_dir=proof_dir,
            repository_root=repo_root,
        )
        == platform_env
    )


def test_process_environment_wins_over_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    _write_env(proof_dir / ".env", "TEST_KEY=dotenv-value\n")
    monkeypatch.setenv("TEST_KEY", "process-value")

    result = load_proof_environment(
        proof_package_dir=proof_dir,
        repository_root=repo_root,
    )

    assert result.loaded is True
    assert os.environ["TEST_KEY"] == "process-value"


def test_dotenv_supplies_missing_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    _write_env(proof_dir / ".env", "MISSING_KEY=from-dotenv\n")
    monkeypatch.delenv("MISSING_KEY", raising=False)

    result = load_proof_environment(
        proof_package_dir=proof_dir,
        repository_root=repo_root,
    )

    assert result.loaded is True
    assert os.environ["MISSING_KEY"] == "from-dotenv"


def test_search_stops_at_repository_root(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    repo_root = parent / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    _write_env(parent / ".env", "OUTSIDE_REPO=1\n")

    assert (
        find_proof_dotenv(
            proof_package_dir=proof_dir,
            repository_root=repo_root,
        )
        is None
    )


def test_missing_dotenv_is_not_an_error(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)

    result = load_proof_environment(
        proof_package_dir=proof_dir,
        repository_root=repo_root,
    )

    assert result.dotenv_path is None
    assert result.loaded is False


def test_proof_package_outside_repository_root_fails(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="proof_package_dir must be"):
        find_proof_dotenv(
            proof_package_dir=outside,
            repository_root=repo_root,
        )


def test_proof_dir_equals_repository_root_finds_root_dotenv(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    root_env = _write_env(repo_root / ".env", "ROOT_ONLY=1\n")

    assert (
        find_proof_dotenv(
            proof_package_dir=repo_root,
            repository_root=repo_root,
        )
        == root_env
    )


def test_working_directory_independence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    proof_env = _write_env(proof_dir / ".env", "PROOF_ONLY=1\n")
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    assert (
        find_proof_dotenv(
            proof_package_dir=proof_dir,
            repository_root=repo_root,
        )
        == proof_env
    )

    result = load_proof_environment(
        proof_package_dir=proof_dir,
        repository_root=repo_root,
    )
    assert result.dotenv_path == proof_env
    assert result.loaded is True


def test_only_nearest_dotenv_file_is_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    _write_env(
        repo_root / ".env",
        "ROOT_ONLY=1\nSHARED=root\n",
    )
    _write_env(
        proof_dir / ".env",
        "PROOF_ONLY=1\nSHARED=proof\n",
    )
    for key in ("ROOT_ONLY", "PROOF_ONLY", "SHARED"):
        monkeypatch.delenv(key, raising=False)

    result = load_proof_environment(
        proof_package_dir=proof_dir,
        repository_root=repo_root,
    )

    assert result.loaded is True
    assert os.environ.get("PROOF_ONLY") == "1"
    assert os.environ.get("SHARED") == "proof"
    assert "ROOT_ONLY" not in os.environ
