# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.proof.intergrax_proof_environment import (
    bootstrap_process_environment,
    find_proof_dotenv,
    load_proof_environment,
    resolve_proof_environment,
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


def test_repo_root_dotenv_loads_intergrax_test_env_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_env(repo_root / ".env", "INTERGRAX_TEST_ENV_VALUE=file\n")
    monkeypatch.delenv("INTERGRAX_TEST_ENV_VALUE", raising=False)

    result = bootstrap_process_environment(
        proof_package_dir=repo_root,
        repository_root=repo_root,
    )

    assert result.loaded is True
    assert os.environ["INTERGRAX_TEST_ENV_VALUE"] == "file"


def test_process_environment_wins_for_intergrax_test_env_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_env(repo_root / ".env", "INTERGRAX_TEST_ENV_VALUE=file\n")
    monkeypatch.setenv("INTERGRAX_TEST_ENV_VALUE", "process")

    result = bootstrap_process_environment(
        proof_package_dir=repo_root,
        repository_root=repo_root,
    )

    assert result.loaded is True
    assert os.environ["INTERGRAX_TEST_ENV_VALUE"] == "process"


def test_resolve_proof_environment_does_not_mutate_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    _write_env(proof_dir / ".env", "INTERGRAX_TEST_PROOF_VALUE=A\n")
    monkeypatch.delenv("INTERGRAX_TEST_PROOF_VALUE", raising=False)

    resolved = resolve_proof_environment(
        proof_package_dir=proof_dir,
        repository_root=repo_root,
        base_environment={"SHARED_OPERATOR": "operator"},
    )

    assert resolved.environment["INTERGRAX_TEST_PROOF_VALUE"] == "A"
    assert resolved.environment["SHARED_OPERATOR"] == "operator"
    assert "INTERGRAX_TEST_PROOF_VALUE" not in os.environ


def test_process_environment_wins_over_proof_dotenv_in_resolved_environment(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    proof_dir = _proof_package(repo_root)
    proof_dir.mkdir(parents=True)
    _write_env(proof_dir / ".env", "INTERGRAX_TEST_PROOF_VALUE=A\n")

    resolved = resolve_proof_environment(
        proof_package_dir=proof_dir,
        repository_root=repo_root,
        base_environment={"INTERGRAX_TEST_PROOF_VALUE": "operator"},
    )

    assert resolved.environment["INTERGRAX_TEST_PROOF_VALUE"] == "operator"


def test_proof_environments_are_independent_from_shared_base(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    proof_a = repo_root / "platform_proofs" / "proof_a"
    proof_b = repo_root / "platform_proofs" / "proof_b"
    proof_a.mkdir(parents=True)
    proof_b.mkdir(parents=True)
    _write_env(proof_a / ".env", "INTERGRAX_TEST_PROOF_VALUE=A\n")
    _write_env(proof_b / ".env", "INTERGRAX_TEST_PROOF_VALUE=B\n")
    base_environment = {"SHARED_OPERATOR": "operator"}

    resolved_a = resolve_proof_environment(
        proof_package_dir=proof_a,
        repository_root=repo_root,
        base_environment=base_environment,
    )
    resolved_b = resolve_proof_environment(
        proof_package_dir=proof_b,
        repository_root=repo_root,
        base_environment=base_environment,
    )

    assert resolved_a.environment["INTERGRAX_TEST_PROOF_VALUE"] == "A"
    assert resolved_b.environment["INTERGRAX_TEST_PROOF_VALUE"] == "B"
    assert resolved_a.environment["SHARED_OPERATOR"] == "operator"
    assert resolved_b.environment["SHARED_OPERATOR"] == "operator"
